package tests.llm;

import static java.lang.System.Logger.Level.DEBUG;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_threads;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppContextState;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParams;

class TravelAgentTests extends AbstractLlmTests {
	private final String systemMsg = "You are a travel agent helping the user to chose the best holiday destination.\n"
			+ "You answer with a city name, and one sentence explanation of your choice, nothing else.";

	private final String userMsg01 = "I want to spend my vacations in Europe.\n"
			+ "I like Italy, but I am open to other destinations, as long as there is nature and culture.\n"
			+ "I have never been to Scandinavia, but it can wait.\n"
			+ "I would like to avoid the usual touristic destinations, so be creative!\n"
			+ "I will travel in autumn, so it should not be too hot.\n"
			+ "Also please consider that I speak French and German in addition to English.\n"
			+ "And I definitely don't like holiday on the beach...";

	private int contextSize = 10240;

	TravelAgentTests(LlamaCppModel model) {
		super(model);
	}

	public void all() throws IOException, InterruptedException {
		// testBaseline();
		testSavedContextState();
	}

	void testBaseline() throws IOException {
		LlamaCppModel model = getModel();

		ContextParams contextParams = LlamaCppContext.defaultContextParams() //
				.with(n_ctx, contextSize) //
				.with(n_threads, getParallelism()) //
		; //

		try (//
				LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			processor.setDebugPrompts(System.err);

			long begin = System.currentTimeMillis();
//			logger.log(DEBUG, SYSTEM.name() + " :\n" + systemMsg);
			processor.write(SYSTEM, systemMsg);

//			logger.log(DEBUG, USER.name() + " :\n" + userMsg01);
			processor.write(USER, userMsg01);

			processor.readMessage(out);

			logger.log(DEBUG, "Dialogue " + (System.currentTimeMillis() - begin) + " ms");
		}
	}

	void testSavedContextState() throws IOException {
		LlamaCppModel model = getModel();

		ContextParams contextParams = LlamaCppContext.defaultContextParams() //
				.with(n_ctx, contextSize) //
				.with(n_threads, getParallelism()) //
		; //

		final LlamaCppContextState savedState;
		final Path sessionFile = Files.createTempFile("jjml_session_", ".llama");
		Runtime.getRuntime().addShutdownHook(new Thread((Runnable) () -> {
			try {
				Files.deleteIfExists(sessionFile);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
		}));

		try (//
				LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			processor.setDebugPrompts(System.err);

			long begin = System.currentTimeMillis();
//			logger.log(DEBUG, SYSTEM.name() + " :\n" + systemMsg);
			processor.write(SYSTEM, systemMsg);

//			logger.log(DEBUG, USER.name() + " :\n" + userMsg01);
			processor.write(USER, userMsg01);

			savedState = new LlamaCppContextState.ByteBufferSavedState();
			logger.log(DEBUG, "Wrote context in " + (System.currentTimeMillis() - begin) + " ms");
			processor.saveContextState(savedState);
			long beginSaveContext = System.currentTimeMillis();
			logger.log(DEBUG, "Saved context in " + (System.currentTimeMillis() - beginSaveContext) + " ms");
			long beginSaveSessionFile = System.currentTimeMillis();
			processor.saveStateFile(sessionFile);
			logger.log(DEBUG, "Saved session file to " + sessionFile + " in "
					+ (System.currentTimeMillis() - beginSaveSessionFile) + " ms");
		}

		String userMsg02 = "Current Date: March 13th 2020.";

		Consumer<LlamaCppInstructProcessor> process = (processor) -> {
//			logger.log(DEBUG, USER.name() + " :\n" + userMsg02);
			processor.write(USER, userMsg02);

//			logger.log(DEBUG, ASSISTANT.name() + " :\n");
			long begin = System.currentTimeMillis();
			try {
				processor.readMessage(out);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
			logger.log(DEBUG, "Generation took " + +(System.currentTimeMillis() - begin) + " ms");
		};

		// deterministic answer
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			long beginLoad = System.currentTimeMillis();
			processor.loadContextState(savedState);
			logger.log(DEBUG, "Loaded context from memory in " + (System.currentTimeMillis() - beginLoad) + " ms");

			process.accept(processor);
		}

		// with temperature
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(true); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			processor.setDebugPrompts(System.err);

			long beginLoad = System.currentTimeMillis();
			processor.loadStateFile(sessionFile);
			logger.log(DEBUG, "Loaded context from file in " + (System.currentTimeMillis() - beginLoad) + " ms");

			process.accept(processor);
		}
	}
}
