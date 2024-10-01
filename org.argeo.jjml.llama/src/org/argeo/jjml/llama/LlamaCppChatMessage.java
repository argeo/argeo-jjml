package org.argeo.jjml.llama;

import java.util.function.Supplier;

/** A message qualified by a role. */
public record LlamaCppChatMessage(String role, String content) {

	public LlamaCppChatMessage(StandardRole role, String content) {
		this(role.get(), content);
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
