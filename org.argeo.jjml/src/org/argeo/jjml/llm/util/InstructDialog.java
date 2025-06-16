package org.argeo.jjml.llm.util;

import java.io.Closeable;
import java.io.IOException;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.DefaultSamplerChainParams;

/**
 * A simple and stable API for the limited but common use case of a dialog
 * between a system ("user") and an LLM ("assistant") trained for instructions
 * ("chat").
 */
public class InstructDialog implements Closeable, Function<String, String> {
	private final LlamaCppContext context;
	private final LlamaCppInstructProcessor processor;

	public InstructDialog(LlamaCppModel model, int contextSize, Path sessionFile, float temperature)
			throws IOException {
		this(model, contextSize, temperature);
		Objects.requireNonNull(sessionFile);

		processor.loadSessionFile(sessionFile);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, String systemPrompt, float temperature)
			throws IOException {
		this(model, contextSize, temperature);
		Objects.requireNonNull(systemPrompt);

		processor.write(InstructRole.SYSTEM, systemPrompt);
	}

	protected InstructDialog(LlamaCppModel model, int contextSize, float temperature) throws IOException {
		Objects.requireNonNull(model);

		ContextParams contextParams = newContextParams().with(ContextParam.n_ctx, contextSize);
		context = new LlamaCppContext(model, contextParams);
		LlamaCppSamplerChain samplerChain = newSamplerChain(context, temperature);
		processor = new LlamaCppInstructProcessor(context, samplerChain);
	}

	/** Writes an input message to an LLM context, and retrieve its output. */
	@Override
	public String apply(String message) {
		Objects.requireNonNull(message);

		processor.write(getInputRole(), message);
		StringWriter sw = new StringWriter();
		try {
			processor.readMessage(sw);
		} catch (IOException e) {
			throw new UncheckedIOException("Cannot read from LLM context", e);
		}
		return sw.toString();
	}

	@Override
	public void close() throws IOException {
		context.close();
	}

	/*
	 * CONTEXT STATE
	 */
	public void saveSessionFile(Path path) {
		Objects.requireNonNull(path);

		processor.saveSessionFile(path);
	}

	/*
	 * DEFAULTS TO BE OVERRIDDEN IF NEEDED
	 */
	protected ContextParams newContextParams() {
		return LlamaCppContext.defaultContextParams();
	}

	protected LlamaCppSamplerChain newSamplerChain(LlamaCppContext context, float temperature) {
		return LlamaCppSamplers.newDefaultSampler(new DefaultSamplerChainParams(temperature));
	}

	protected Supplier<String> getSystemRole() {
		return InstructRole.SYSTEM;
	}

	protected Supplier<String> getInputRole() {
		return InstructRole.USER;
	}

	protected Supplier<String> getOutputRole() {
		return InstructRole.ASSISTANT;
	}

}
