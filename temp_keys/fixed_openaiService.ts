// Direct OpenAI integration service
// This service communicates directly with the OpenAI API

// OpenAI API key from your config
const OPENAI_API_KEY = "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";

// Trading assistant system prompt
const SYSTEM_PROMPT = `You are BenBot, an AI assistant for a trading bot system.`;

// Initialize conversation history
let conversationHistory: Array<{ role: string; content: string }> = [];

/**
 * Gets a response from OpenAI
 * @param message - User message
 * @returns AI response
 */
export const getOpenAIResponse = async (message: string): Promise<string> => {
  try {
    // Add user message to history
    conversationHistory.push({ role: "user", content: message });
    
    // Keep only the last 10 messages to avoid token limits
    if (conversationHistory.length > 10) {
      conversationHistory = conversationHistory.slice(conversationHistory.length - 10);
    }
    
    // Prepare messages
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      ...conversationHistory
    ];
    
    // Call OpenAI API
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4-turbo",
        messages: messages,
        temperature: 0.7,
        max_tokens: 800
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`OpenAI API error: ${errorData.error?.message || 'Unknown error'}`);
    }
    
    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    
    // Add AI response to history
    conversationHistory.push({ role: "assistant", content: aiResponse });
    
    return aiResponse;
  } catch (error) {
    console.error("Error calling OpenAI:", error);
    return "I'm having trouble connecting to my AI services right now. Please try again later.";
  }
};

/**
 * Reset conversation history
 */
export const resetConversation = (): void => {
  conversationHistory = [];
};

/**
 * Set conversation history
 */
export const setConversationHistory = (history: Array<{ role: string; content: string }>): void => {
  conversationHistory = history;
};

// Ensure this file is treated as a module
export {};
