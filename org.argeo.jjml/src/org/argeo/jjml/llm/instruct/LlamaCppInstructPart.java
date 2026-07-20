package org.argeo.jjml.llm.instruct;

public interface LlamaCppInstructPart {
	String getRole();

	Iterable<LlamaCppInstructBlock<?>> getBlocks();
}
