import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines multiple class values into a single className string,
 * with proper handling of Tailwind CSS classes.
 * 
 * @param inputs - Class values to be combined
 * @returns - Merged className string
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
