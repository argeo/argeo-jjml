package org.argeo.jjml.llm.util;

import java.io.IOException;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.function.Consumer;
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
public class InstructDialog implements AutoCloseable, Function<String, String>, Consumer<String> {
	private final LlamaCppContext context;
	private final LlamaCppInstructProcessor processor;

	/*
	 * CONSTRUCTORS
	 */
	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, String systemPrompt) {
		this(model, contextSize, parallelism, systemPrompt, 0);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, Path stateFile) throws IOException {
		this(model, contextSize, parallelism, stateFile, 0);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, String systemPrompt,
			float temperature) {
		this(model, contextSize, parallelism, temperature);
		processor.write(getSystemRole(), systemPrompt);
	}

	public InstructDialog(LlamaCppModel model, int contextSize, int parallelism, Path stateFile, float temperature)
			throws IOException {
		this(model, contextSize, parallelism, temperature);
		processor.loadStateFile(stateFile);
	}

	protected InstructDialog(LlamaCppModel model, int contextSize, int parallelism, float temperature) {
		ContextParams contextParams = newContextParams() //
				.with(ContextParam.n_ctx, contextSize) //
				.with(ContextParam.n_threads, parallelism) //
		;

		context = new LlamaCppContext(model, contextParams);
		LlamaCppSamplerChain samplerChain = newSamplerChain(context, temperature);
		processor = new LlamaCppInstructProcessor(context, samplerChain);
	}

	/** Writes an input message to an LLM context, and retrieve its output. */
	@Override
	public String apply(String message) {
		processor.write(getInputRole(), message);
		StringWriter sw = new StringWriter();
		try {
			processor.readMessage(sw);
		} catch (IOException e) {
			throw new UncheckedIOException("Cannot read from LLM context", e);
		}
		return sw.toString();
	}

	/**
	 * Appends an input to the context ("user message"), without triggering
	 * generation from the model.
	 */
	@Override
	public void accept(String message) {
		processor.write(getInputRole(), message);
	}

	/** Free context resources. */
	@Override
	public void close() throws IOException {
		context.close();
	}

	/*
	 * CONTEXT STATE
	 */
	public void saveStateFile(Path path) throws IOException {
		processor.saveStateFile(path);
	}

	/*
	 * DEFAULTS TO BE OVERRIDDEN IF NEEDED
	 */
	/**
	 * The context parameters to use when initializing. Default implementation uses
	 * {@link LlamaCppContext#defaultContextParams()}.
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

}
