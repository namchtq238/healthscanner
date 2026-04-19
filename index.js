require('dotenv').config();

const {
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  Client,
  GatewayIntentBits,
} = require('discord.js');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const ANALYSIS_PROMPT =
  'Analyze this food image. Return a raw JSON object with these exact keys: calories, protein, carbs, fat, food_name. Ensure values are numbers (except food_name). No markdown, no extra text.';
const MODEL_NAME = 'gemini-1.5-flash';
const MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024;
const REQUIRED_KEYS = ['calories', 'protein', 'carbs', 'fat', 'food_name'];
const SUPPORTED_IMAGE_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/heic',
  'image/heif',
]);

const { DISCORD_TOKEN, GEMINI_API_KEY } = process.env;

if (!DISCORD_TOKEN || !GEMINI_API_KEY) {
  console.error('Missing DISCORD_TOKEN or GEMINI_API_KEY in .env');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: MODEL_NAME });

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once('ready', () => {
  console.log(`Logged in as ${client.user.tag}`);
});

client.on('messageCreate', async (message) => {
  if (message.author.bot) {
    return;
  }

  const imageAttachment = getFirstImageAttachment(message.attachments);
  if (!imageAttachment) {
    return;
  }

  try {
    const imageData = await downloadAttachment(imageAttachment);
    const nutrition = await analyzeFoodImage(imageData.buffer, imageData.mimeType);
    const shortcutUrl = buildShortcutUrl(nutrition);
    const replyContent = formatNutritionSummary(nutrition);
    const components = [buildShortcutButton(shortcutUrl)];

    await message.reply({
      content: replyContent,
      components,
      allowedMentions: { repliedUser: true },
    });
  } catch (error) {
    console.error('Failed to process image:', error);

    await safeReply(
      message,
      'I could not analyze that image. Make sure it is a valid food photo in JPEG, PNG, WEBP, HEIC, or HEIF format, then try again.'
    );
  }
});

client.on('error', (error) => {
  console.error('Discord client error:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});

client.login(DISCORD_TOKEN);

function getFirstImageAttachment(attachments) {
  return attachments.find((attachment) => {
    const mimeType = normalizeMimeType(attachment.contentType, attachment.name);
    return Boolean(mimeType);
  });
}

async function downloadAttachment(attachment) {
  const mimeType = normalizeMimeType(attachment.contentType, attachment.name);

  if (!mimeType) {
    throw new Error('Unsupported or missing image content type.');
  }

  if (attachment.size > MAX_INLINE_IMAGE_BYTES) {
    throw new Error('Image is larger than Gemini inline upload limits.');
  }

  const response = await fetch(attachment.url);

  if (!response.ok) {
    throw new Error(`Failed to download image: ${response.status} ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  if (buffer.length === 0) {
    throw new Error('Downloaded image buffer is empty.');
  }

  if (buffer.length > MAX_INLINE_IMAGE_BYTES) {
    throw new Error('Downloaded image is larger than Gemini inline upload limits.');
  }

  return { buffer, mimeType };
}

async function analyzeFoodImage(buffer, mimeType) {
  const result = await model.generateContent({
    contents: [
      {
        role: 'user',
        parts: [
          {
            inlineData: {
              data: buffer.toString('base64'),
              mimeType,
            },
          },
          {
            text: ANALYSIS_PROMPT,
          },
        ],
      },
    ],
    generationConfig: {
      responseMimeType: 'application/json',
      temperature: 0.2,
    },
  });

  const responseText = result.response.text();
  return parseNutritionPayload(responseText);
}

function parseNutritionPayload(rawText) {
  const sanitized = stripCodeFences(String(rawText || '').trim());
  const candidates = [sanitized, extractFirstJsonObject(sanitized)].filter(Boolean);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      return normalizeNutritionObject(parsed);
    } catch (error) {
      continue;
    }
  }

  throw new Error(`Unable to parse Gemini response as JSON: ${rawText}`);
}

function normalizeNutritionObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Gemini response must be a JSON object.');
  }

  const normalized = {};

  for (const key of REQUIRED_KEYS) {
    if (!(key in value)) {
      throw new Error(`Missing required key: ${key}`);
    }

    if (key === 'food_name') {
      const foodName = String(value[key]).trim();
      if (!foodName) {
        throw new Error('food_name must be a non-empty string.');
      }

      normalized[key] = foodName;
      continue;
    }

    const numberValue =
      typeof value[key] === 'number' ? value[key] : Number(String(value[key]).trim());

    if (!Number.isFinite(numberValue)) {
      throw new Error(`${key} must be a finite number.`);
    }

    normalized[key] = roundToTwo(numberValue);
  }

  return normalized;
}

function buildShortcutUrl(nutrition) {
  const input = encodeURIComponent(JSON.stringify(nutrition));
  return `shortcuts://run-shortcut?name=input-health&input=${input}`;
}

function buildShortcutButton(url) {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setLabel('\u{1F4E5} Log to Apple Health')
      .setStyle(ButtonStyle.Link)
      .setURL(url)
  );
}

function formatNutritionSummary(nutrition) {
  return [
    `**${nutrition.food_name}**`,
    `Calories: ${formatNumber(nutrition.calories)}`,
    `Protein: ${formatNumber(nutrition.protein)} g`,
    `Carbs: ${formatNumber(nutrition.carbs)} g`,
    `Fat: ${formatNumber(nutrition.fat)} g`,
  ].join('\n');
}

async function safeReply(message, content) {
  try {
    await message.reply({
      content,
      allowedMentions: { repliedUser: true },
    });
  } catch (replyError) {
    console.error('Failed to send Discord reply:', replyError);
  }
}

function stripCodeFences(text) {
  return text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
}

function extractFirstJsonObject(text) {
  const start = text.indexOf('{');
  if (start === -1) {
    return null;
  }

  let depth = 0;
  let inString = false;
  let isEscaped = false;

  for (let index = start; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (isEscaped) {
        isEscaped = false;
        continue;
      }

      if (char === '\\') {
        isEscaped = true;
        continue;
      }

      if (char === '"') {
        inString = false;
      }

      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === '{') {
      depth += 1;
      continue;
    }

    if (char === '}') {
      depth -= 1;

      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }

  return null;
}

function normalizeMimeType(contentType, fileName = '') {
  const normalizedType = String(contentType || '')
    .split(';')[0]
    .trim()
    .toLowerCase();

  if (SUPPORTED_IMAGE_TYPES.has(normalizedType)) {
    return normalizedType;
  }

  const extension = fileName.split('.').pop()?.toLowerCase();

  switch (extension) {
    case 'jpg':
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'webp':
      return 'image/webp';
    case 'heic':
      return 'image/heic';
    case 'heif':
      return 'image/heif';
    default:
      return null;
  }
}

function roundToTwo(value) {
  return Math.round(value * 100) / 100;
}

function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
