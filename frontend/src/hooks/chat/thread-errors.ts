function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function isMissingThreadError(
  error: unknown,
  threadId: string | null
): boolean {
  if (!threadId) {
    return false;
  }

  const message = errorMessage(error);
  if (!message.includes("404")) {
    return false;
  }
  if (!message.includes(threadId)) {
    return false;
  }

  return message.toLowerCase().includes("not found");
}
