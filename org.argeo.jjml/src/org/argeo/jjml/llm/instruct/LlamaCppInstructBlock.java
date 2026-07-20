package org.argeo.jjml.llm.instruct;

public interface LlamaCppInstructBlock<T> {
	/**
	 * If the type is <code>null</code>, the content is expected to be plain text
	 * (either a {@link CharSequence} or, if not, {@link Object#toString()} of the
	 * content).
	 */
	String getType();

	/** If <code>null</code>, this block is empty. */
	T getContent();
}
