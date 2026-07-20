package org.argeo.jjml.llm.instruct;

import java.util.function.Supplier;

public class LlamaCppTextBlock implements LlamaCppInstructBlock<CharSequence> {
	/** can be null (default text) */
	private final String type;

	/** can be null (empty) */
	private final CharSequence content;

	public LlamaCppTextBlock(CharSequence content) {
		this(content, null);
	}

	public LlamaCppTextBlock(CharSequence content, Supplier<String> type) {
		this.type = type.get();
		this.content = content;
	}

	@Override
	public String getType() {
		return type;
	}

	@Override
	public CharSequence getContent() {
		return content;
	}

	@Override
	public String toString() {
		return getContent().toString();
	}

}
