package org.argeo.jjml.llama;

import java.util.function.Supplier;

/** A message qualified by a role. */
public class LlamaCppChatMessage {
	private final String role;
	private final String content;

	public LlamaCppChatMessage(String role, String content) {
		this.role = role;
		this.content = content;
	}

	public LlamaCppChatMessage(Supplier<String> role, String content) {
		this(role.get(), content);
	}

	public String getRole() {
		return role;
	}

	public String getContent() {
		return content;
	}
}
