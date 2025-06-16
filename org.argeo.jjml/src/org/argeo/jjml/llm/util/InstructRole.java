package org.argeo.jjml.llm.util;

import java.util.function.Supplier;

/** Commonly used instruct roles. */
public enum InstructRole implements Supplier<String> {
	SYSTEM("system"), //
	USER("user"), //
	ASSISTANT("assistant"), //
	;

	private final String role;

	private InstructRole(String role) {
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
