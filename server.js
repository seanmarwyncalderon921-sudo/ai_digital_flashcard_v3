const express = require('express');
const dotenv = require('dotenv');
const { Groq } = require('groq-sdk');
const sharp = require('sharp');

dotenv.config();

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

function missingKeyError(keyName) {
  const error = new Error(`${keyName} is missing. Add it to .env and restart the server.`);
  error.statusCode = 503;
  return error;
}

function parseFlashcards(responseText) {
  const data = JSON.parse(responseText.trim());
  if (!Array.isArray(data.cards)) {
    throw new Error('The AI response did not include a cards array.');
  }
  data.cards = data.cards.map(card => {
    if (typeof card.question !== 'string' || typeof card.answer !== 'string') return card;

    let question = card.question;
    const answerItems = card.answer.split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => /^\d+[.)]\s+/.test(line));
    if (answerItems.length >= 2) {
      const questionLines = question.split(/\r?\n/);
      const firstAnswerItem = questionLines.findIndex(line => line.trim() === answerItems[0]);
      if (firstAnswerItem >= 0) {
        const duplicatedItems = questionLines.slice(firstAnswerItem, firstAnswerItem + answerItems.length);
        if (answerItems.every((item, index) => duplicatedItems[index]?.trim() === item)) {
          const promptLines = questionLines.slice(0, firstAnswerItem);
          while (promptLines.length && /^(?:answer|functions|these are|they are)\s*:?\s*$/i.test(promptLines[promptLines.length - 1].trim())) {
            promptLines.pop();
          }
          question = promptLines.join('\n').trim() || question;
        }
      }
    }

    const normalizedAnswer = card.answer.trim().replace(/[.!?]+$/, '').toLowerCase();
    const questionHasChoices = question.split(/\r?\n/).some(line => /^\s*[A-D][.)]\s+/.test(line));
    const type = /^(true|false)$/.test(normalizedAnswer)
      ? 'true-false'
      : questionHasChoices
        ? 'multiple-choice'
        : answerItems.length >= 2
          ? 'enumeration'
          : 'identification';

    return { ...card, question, type };
  });
  return data;
}

async function generateWithGroq(messages, temperature = 0.5) {
  if (!process.env.GROQ_API_KEY) throw missingKeyError('GROQ_API_KEY');
  const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  const completion = await groq.chat.completions.create({
    messages,
    model: process.env.GROQ_MODEL || 'qwen/qwen3.8-27b',
    response_format: { type: 'json_object' },
    temperature
  });
  const responseText = completion.choices?.[0]?.message?.content;
  if (!responseText) throw new Error('Groq returned an empty response.');
  return parseFlashcards(responseText);
}

app.post('/api/flashcards', async (req, res) => {
  const { notes, count, images, image, mimeType, gradeLevel, academicStrand, difficulty } = req.body;

  try {
    const parsedCount = Number.parseInt(count, 10);
    const requestedCount = Number.isInteger(parsedCount) && parsedCount > 0
      ? Math.min(parsedCount, 50)
      : 10;
    const requestedDifficulty = String(difficulty || 'medium').toLowerCase();
    const difficultyLevel = ['easy', 'medium', 'hard'].includes(requestedDifficulty)
      ? requestedDifficulty
      : 'medium';
    const difficultyGuidance = {
      easy: 'Focus on foundational facts, direct recall, and clear wording.',
      medium: 'Use a balanced mix of recall and understanding appropriate to the target grade.',
      hard: 'Focus on detailed understanding, application, and connections between concepts.'
    }[difficultyLevel];
    const difficultyCardMix = {
      easy: 'Favor Multiple Choice and True/False (about 40% each), with the remaining cards as Identification. Do not use Enumeration.',
      medium: 'Use Multiple Choice, True/False, and Identification in roughly equal shares. Do not force Enumeration into this level.',
      hard: 'Keep the deck varied: use Enumeration for about one-third of cards, and split the remaining cards as evenly as possible among Multiple Choice, True/False, and Identification. Include all four formats when at least four cards are requested; Enumeration must not dominate the deck.'
    }[difficultyLevel];
    const diversityGuidance = requestedCount >= 20
      ? 'This is a large deck. Before drafting, make a private coverage list of distinct facts from every note and image, then assign one different fact or concept to each card. Do not test the same fact twice in different formats. Vary both the subject matter and question wording.'
      : 'Give each card a distinct fact or concept. Avoid duplicate facts and vary question wording.';

    let systemInstructions = "You are an expert academic reviewer. Generate " + requestedCount + " structured study flashcards when the source supports that many unique facts.\n" +
                             "Target Level: " + (gradeLevel || 'Senior High School') + "\n" +
                             "Target Track/Strand: " + (academicStrand || 'General') + "\n\n" +
                             "Target Difficulty: " + difficultyLevel + ". " + difficultyGuidance + "\n\n" +
                             "Card type mix for this difficulty: " + difficultyCardMix + "\n\n" +
                             "Coverage and variety: " + diversityGuidance + " If the material does not support " + requestedCount + " distinct, accurate cards, return fewer cards rather than repeat facts or invent content.\n\n" +
                             "CRITICAL FORMAT RULES:\n" +
                             "1. Follow the card type mix for the selected difficulty. Do not force every card type into every deck when it conflicts with that mix.\n" +
                             "2. For Multiple Choice, put the question and options A, B, C, and D in the \"question\" string on separate lines, and make \"answer\" the correct option string.\n" +
                             "3. For Identification, make \"question\" a clear definition and \"answer\" the exact term.\n" +
                             "4. For Enumeration, keep the \"question\" to one direct request only, such as \"List three specific functions performed by the presentation layer of the OSI model.\" Put the numbered items only in \"answer\". Never include answer items, hints, or a list preview in \"question\".\n" +
                             "5. For True/False, write the question as \"True or False: [one factual statement]\", make \"answer\" exactly \"True\" or \"False\", and add a concise \"explanation\" that justifies the verdict.\n" +
                             "6. Use all provided text and images together as source material. Generate exactly the requested total number of cards; do not make only one card per image. Each image may support several distinct questions.\n" +
                             "7. Cover different facts and concepts from across the material. Do not repeat an answer, fact, or question template; vary wording and task focus while staying faithful to the source.\n" +
                             "8. For every card, the \"question\" field must contain only the prompt and must never reveal or repeat its answer.\n\n" +
                             "You must respond with a pure, raw JSON object containing a single top-level array property called \"cards\".\n" +
                             "Each item must contain string properties \"question\" and \"answer\". True/False items must also contain a string property called \"explanation\". Do not add other properties.\n" +
                             "Do not enclose your output in markdown code blocks like ```json. Return only valid, minified raw JSON text.";

    const messages = [
      { role: "system", content: systemInstructions }
    ];

    const inputImages = Array.isArray(images)
      ? images
      : image && mimeType ? [{ data: image, mimeType }] : [];
    const validImages = inputImages.filter(file => file && typeof file.data === 'string');
    const normalizedImages = await Promise.all(validImages.map(async (file, index) => {
      try {
        const jpegBuffer = await sharp(Buffer.from(file.data, 'base64'))
          .rotate()
          .jpeg({ quality: 90 })
          .toBuffer();
        return { data: jpegBuffer.toString('base64'), mimeType: 'image/jpeg' };
      } catch {
        const fileName = file.name ? `"${file.name}"` : `number ${index + 1}`;
        const error = new Error(`Could not read image ${fileName}. Supported formats include JPEG, PNG, WebP, TIFF, GIF, SVG, and AVIF.`);
        error.statusCode = 400;
        throw error;
      }
    }));

    if (normalizedImages.length > 0) {
      const content = [
        { type: "text", text: "Analyze these study note images along with any text inputs provided: " + (notes || "") }
      ];
      normalizedImages.forEach(file => {
        content.push({
          type: "image_url",
          image_url: { url: `data:${file.mimeType};base64,${file.data}` }
        });
      });
      messages.push({ role: "user", content });
    } else {
      messages.push({
        role: "user",
        content: "Here is the raw text study material: " + (notes || "Generate default relevant track items.")
      });
    }

    const parsedData = await generateWithGroq(messages, requestedCount >= 20 ? 0.65 : 0.5);
    res.json(parsedData);

  } catch (error) {
    console.error("Flashcard generation error:", error.message);
    res.status(error.statusCode || 500).json({ error: error.message || "Failed to process flashcard data via Groq." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("Server running at http://localhost:" + PORT);
});
