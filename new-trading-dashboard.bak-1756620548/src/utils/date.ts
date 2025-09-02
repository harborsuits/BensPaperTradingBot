/**
 * Formats a date to a relative time string (e.g., "3 hours ago", "Just now", etc.)
 * @param date The date to format
 * @returns A string representing the relative time
 */
export function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  
  // Convert ms to seconds
  const seconds = Math.floor(diff / 1000);
  
  if (seconds < 60) {
    return 'Just now';
  }
  
  // Convert to minutes
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
  }
  
  // Convert to hours
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
  }
  
  // Convert to days
  const days = Math.floor(hours / 24);
  if (days < 30) {
    return `${days} day${days !== 1 ? 's' : ''} ago`;
  }
  
  // Convert to months
  const months = Math.floor(days / 30);
  if (months < 12) {
    return `${months} month${months !== 1 ? 's' : ''} ago`;
  }
  
  // Convert to years
  const years = Math.floor(months / 12);
  return `${years} year${years !== 1 ? 's' : ''} ago`;
}

/**
 * Formats a date string to a human readable format
 * @param dateString The date string to format
 * @param includeTime Whether to include the time in the format
 * @returns A formatted date string
 */
export function formatDate(dateString: string, includeTime: boolean = false): string {
  const date = new Date(dateString);
  
  const options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  };
  
  if (includeTime) {
    options.hour = '2-digit';
    options.minute = '2-digit';
    options.hour12 = true;
  }
  
  return date.toLocaleDateString('en-US', options);
}

/**
 * Get a short formatted date (MM/DD)
 * @param date The date to format
 * @returns A short formatted date string
 */
export function getShortDate(date: Date): string {
  const month = date.getMonth() + 1;
  const day = date.getDate();
  
  return `${month}/${day}`;
}
