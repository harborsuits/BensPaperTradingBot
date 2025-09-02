// Direct OpenAI integration for the dashboard
// This allows BenBot to use real AI without requiring the backend API

// Your API key (from the config)
const OPENAI_API_KEY = "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";

// System prompt similar to what the BenBotAssistant uses
const SYSTEM_PROMPT = `You are BenBot, an AI assistant for a trading bot system.`;

/**
 * Direct integration with OpenAI API
 * @param message The user's message
 * @returns Promise with AI response
 */
export const getDirectResponse = async (message: string): Promise<string> => {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4-turbo",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: message }
        ],
        temperature: 0.7,
        max_tokens: 500
      })
    });
    
    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error("Error calling OpenAI directly:", error);
    return "I'm having trouble connecting to OpenAI right now. Please try again later.";
  }
};

// Ensure this file is treated as a module
export {};
