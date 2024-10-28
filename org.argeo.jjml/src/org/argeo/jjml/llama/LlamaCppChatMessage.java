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

	public LlamaCppChatMessage(StandardRole role, String content) {
		this(role.get(), content);
	}

	public String getRole() {
		return role;
	}

	public String getContent() {
		return content;
	}

	/** Commonly used roles, for convenience. */
	public static enum StandardRole implements Supplier<String> {
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

	}
}
