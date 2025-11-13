package org.argeo.jjml.llm.util;

import java.io.IOException;
import java.nio.file.Path;

import org.argeo.jjml.llm.LlamaCppModel;

/** @deprecated Use {@link InstructDialogue} instead. */
@Deprecated
public class InstructDialog extends InstructDialogue {

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, float temperature) {
		super(model, contextSize, parallelism, temperature);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, Path stateFile, float temperature)
			throws IOException {
		super(model, contextSize, parallelism, stateFile, temperature);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, Path stateFile) throws IOException {
		super(model, contextSize, parallelism, stateFile);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, String systemPrompt,
			float temperature) {
		super(model, contextSize, parallelism, systemPrompt, temperature);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, String systemPrompt) {
		super(model, contextSize, parallelism, systemPrompt);
	}

}
