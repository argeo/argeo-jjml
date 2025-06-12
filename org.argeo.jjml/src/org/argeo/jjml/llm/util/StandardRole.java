package org.argeo.jjml.llm.util;

import java.util.function.Supplier;

import org.argeo.jjml.llm.LlamaCppChatMessage;

/** Commonly used instruct roles. */
public enum StandardRole implements Supplier<String> {
	SYSTEM("system"), //
	USER("user"), //
	ASSISTANT("assistant"), //
	;

	private final String role;

	private StandardRole(String role) {
		this.role = role;
	}

	@Override
	public String get() {
		return role;
	}

	@Override
	public String toString() {
		return get();
	}

	/** Creates a new chat message with this role and the provided text. */
	public LlamaCppChatMessage msg(String text) {
		return new LlamaCppChatMessage(this, text);
	}
}
