package org.argeo.jjml.llm;

import java.util.Collections;
import java.util.function.Supplier;

import org.argeo.jjml.llm.instruct.LlamaCppInstructBlock;
import org.argeo.jjml.llm.instruct.LlamaCppInstructPart;
import org.argeo.jjml.llm.instruct.LlamaCppTextBlock;

/** A message qualified by a role. */
@Deprecated
public class LlamaCppChatMessage implements LlamaCppInstructPart {
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

	@Override
	public Iterable<LlamaCppInstructBlock<?>> getBlocks() {
		return Collections.singleton(new LlamaCppTextBlock(content));
	}

	@Override
	public String toString() {
		return content;
	}
	
	
}
