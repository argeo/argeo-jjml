import static java.lang.System.Logger.Level.DEBUG;
import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llm.LlamaCppSamplers.newJavaSampler;
import static org.argeo.jjml.llm.params.ContextParam.embeddings;
import static org.argeo.jjml.llm.params.ContextParam.kv_unified;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_threads;
import static org.argeo.jjml.llm.params.ContextParam.n_ubatch;
import static org.argeo.jjml.llm.util.InstructRole.ASSISTANT;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppContextState;
import org.argeo.jjml.llm.LlamaCppEmbeddingProcessor;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppJavaSampler;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppNativeSampler;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.LlamaCppTextProcessor;
import org.argeo.jjml.llm.LlamaCppVocabulary;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.ModelParams;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
class JjmlSmokeTests {
	/** Default location for GGUF model files. */
	private final static Path MODELS_BASE = File.separatorChar == '/'
			? Paths.get(System.getProperty("user.home"), ".cache", "llama.cpp")
			: Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");

	private final static Logger logger = System.getLogger(JjmlSmokeTests.class.getName());

	private int parallelism = Runtime.getRuntime().availableProcessors();

	public void main(List<String> args) throws Exception, AssertionError {
		try {
			if (!getClass().desiredAssertionStatus()) {
				logger.log(ERROR, "Assertions must be enabled. Please call Java with the -ea option.");
				return;
			}

			// even without a model we can check whether native libraries are loading
			assert ((BooleanSupplier) () -> {
				LlamaCppNative.ensureLibrariesLoaded();
				return true;
			}).getAsBoolean();
			logger.log(INFO, "Native libraries properly loaded.");

			if (args.isEmpty()) {
				logger.log(ERROR, "Usage: " + getClass().getSimpleName() + " <path to GGUF model>");
				return;
			}

			Path modelPath = Paths.get(args.get(0));
			if (!Files.exists(modelPath)) {
				String hfRepo = args.get(0);
				String quantization = "Q4_K_M";
				if (hfRepo.contains(":")) {
					quantization = hfRepo.split(":")[1];
					hfRepo = hfRepo.split(":")[0];
				}
				String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
				String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
				modelPath = MODELS_BASE.resolve(localFileName);

			}
			if (!Files.exists(modelPath))
				throw new IllegalArgumentException("Could not find GGUF model " + modelPath);

			ModelParams modelParams = defaultModelParams();
			logger.log(INFO, "Loading model " + modelPath + " ...");
			Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams,
					new LoadModelProgressCallback(), null);
			try (LlamaCppModel model = loaded.get();) {
				logger.log(INFO, "Model " + model.getDescription());
				logger.log(INFO, model.getLayerCount() + " layers");
				logger.log(INFO, model.getEmbeddingSize() + " embedding size");
				logger.log(INFO, model.getVocabularySize() + " vocabulary size");
				logger.log(INFO, model.getContextTrainingSize() + " context training size");
				StringBuilder sb = new StringBuilder();
				for (String key : model.getMetadata().keySet())
					sb.append(key + "=" + model.getMetadata().get(key) + "\n");
				logger.log(DEBUG, "Metadata:\n" + sb);

				assertVocabulary(model.getVocabulary());
				// TODO return if vocabulary only
//				if (true)
//					return;

				assertLoadUnloadDefaultContext(model);
				assertEmbeddings(model);
				assertBatch(model);
				assertJavaSampler(model);
				assertChat(model);
				assertSavedContextState(model);
			}
		} catch (Exception | AssertionError e) {
			logger.log(Level.ERROR, "Smoke tests failed", e);
			throw e;
		} finally {
			LlamaCppBackend.destroy();
		}
	}

	void assertVocabulary(LlamaCppVocabulary vocabulary) {
		int size = 256;

		// in direct, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in array, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in string, out direct
		assertVocabulary(vocabulary, //
				null, //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in direct, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in array, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in string, out array
		assertVocabulary(vocabulary, //
				null, //
				IntBuffer.allocate(size / Integer.BYTES));
	}

	void assertVocabulary(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer out) {
		assert testTokenizeDetokenize(vocabulary, in, out, "Hello World!");
		assert testTokenizeDetokenize(vocabulary, in, out, "Même si je suis Français, je dis bonjour au monde");
		assert testTokenizeDetokenize(vocabulary, in, out, "ἔορθoι χθόνιοι"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "السلام عليكم"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "¡Hola и أَشْكَرُ мир! 👋🏼🌍");
		logger.log(INFO, "Vocabulary smoke tests variant PASSED");
	}

	boolean testTokenizeDetokenize(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer buf, String msg) {
		if (in != null)
			in.clear();
		buf.clear();

		logger.log(DEBUG, msg);
		if (in == null) {
			IntBuffer tokens = vocabulary.tokenize(msg);
			buf.put(tokens);
		} else {
			in.put(msg.getBytes(UTF_8));
			in.flip();
			vocabulary.tokenize(msg, buf);
		}
		buf.flip();
		logger.log(DEBUG, logIntegers(buf, 32, ", "));
		String str;
		if (in == null) {
			str = vocabulary.deTokenize(buf);
		} else {
			in.clear();
			vocabulary.deTokenize(buf, in);
			in.flip();
			str = UTF_8.decode(in).toString();
		}
		assert str.equals(msg);
		return true;
	}

	void assertLoadUnloadDefaultContext(LlamaCppModel model) {
		try (LlamaCppContext context = new LlamaCppContext(model);) {
			assert context.getContextSize() > 0;
		}
		logger.log(INFO, "Load default context smoke tests PASSED");
	}

	void assertEmbeddings(LlamaCppModel model) {
		int batchSize = 512;
		try (LlamaCppContext context = new LlamaCppContext(model, LlamaCppContext.defaultContextParams() //
				.with(embeddings, true) //
				.with(n_ctx, 6144) //
				.with(n_batch, batchSize) //
				.with(n_ubatch, batchSize) // must be same for embeddings
				.with(kv_unified, true) // required for robustness
		);) {
			LlamaCppEmbeddingProcessor embeddingProcessor = new LlamaCppEmbeddingProcessor(context);

			List<String> prompts = new ArrayList<>();
			prompts.add("Hello world!");
			prompts.add("Good night and good luck.");
			for (String s : prompts)
				logger.log(DEBUG, "=>\n" + s);

			float[][] embeddings = embeddingProcessor.processEmbeddings(prompts);
			assert embeddings.length != 0;

			for (float[] embedding : embeddings) {
				logger.log(DEBUG, "<=\n[ " + embedding[0] + ", " + embedding[1] + ", ... ]");
			}
		}
		logger.log(INFO, "Embeddings smoke tests PASSED");
	}

	void assertBatch(LlamaCppModel model) {
		String prompt = "Write HELLO\n"//
				+ "HELLO\n"//
				+ "Write WORLD\n"//
				+ "WORLD\n"//
				+ "Write TEST\n" //
		;

		// !! max seq_id must be < 64
		// TODO understand why
		Integer[] sequenceIds = { 1, 10, 63 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * prompt.length()) //
						.with(kv_unified, true) // required for robustness
				); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			System.out.println("=>\n" + prompt);
			String str = processor.processBatch(prompt);
			System.out.println("<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
		logger.log(INFO, "Batch smoke tests PASSED");
	}

	void assertJavaSampler(LlamaCppModel model) {
		Integer[] sequenceIds = { 1 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * 64) //
						.with(kv_unified, true) // required for robustness
				); //
				LlamaCppSamplerChain chain = new LlamaCppSamplerChain(
						newJavaSampler(new LlamaCppJavaSampler.SimpleGreedy())); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			String prompt = "Write HELLO\n"//
					+ "Hello\n"//
					+ "Write World\n"//
					+ "WORLD\n"//
					+ "Write test\n" //
			;
			System.out.println("=>\n" + prompt);
			String str = processor.processBatch(prompt);
			System.out.println("<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
		logger.log(INFO, "Java sampler smoke tests PASSED");
	}

	void assertChat(LlamaCppModel model) throws IOException {
		try (//
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 20480) //
						.with(n_batch, 1024) //
						.with(n_threads, parallelism) //
				); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);

			String systemMsg = "You are a helpful assistant, which answers as briefly as possible.";
			System.out.println(SYSTEM.name() + " :\n" + systemMsg);
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "Introduce the Java programming language in no more than two sentences.";
			System.out.println(USER.name() + " :\n" + userMsg01);
			processor.write(USER, userMsg01);

			System.out.println(ASSISTANT.name() + " :\n");
			processor.readMessage(System.out);

			// make sure it can deal with a second message
			String userMsg02 = "Thank you!";
			System.out.println(USER.name() + " :\n" + userMsg02);
			processor.write(USER, userMsg02);

			System.out.println(ASSISTANT.name() + " :\n");
			processor.readMessage(System.out);
		}
		logger.log(INFO, "Chat smoke tests PASSED");
	}

	void assertSavedContextState(LlamaCppModel model) throws IOException {
		ContextParams contextParams = LlamaCppContext.defaultContextParams() //
				.with(n_ctx, 20480) //
				.with(n_batch, 1024) //
				.with(n_threads, parallelism) //
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

			long begin = System.currentTimeMillis();
			String systemMsg = "You are a travel agent helping the user to chose the best holiday destination.\n"
					+ "You answer with a city name, and one sentence explanation of your choice, nothing else.";
			System.out.println(SYSTEM.name() + " :\n" + systemMsg);
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "I want to spend my vacations in Europe.\n"
					+ "I like Italy, but I am open to other destinations, as long as there is nature and culture.\n"
					+ "I have never been to Scandinavia, but it can wait.\n"
					+ "I would like to avoid the usual touristic destinations, so be creative!\n"
					+ "I will travel in autumn, so it should not be too hot.\n"
					+ "Also please consider that I speak French and German in addition to English.\n"
					+ "And I definitely don't like holiday on the beach...";
			System.out.println(USER.name() + " :\n" + userMsg01);
			processor.write(USER, userMsg01);

			savedState = new LlamaCppContextState.ByteBufferSavedState();
			logger.log(INFO, "Wrote context in " + (System.currentTimeMillis() - begin) + " ms");
			processor.saveContextState(savedState);
			long beginSaveContext = System.currentTimeMillis();
			logger.log(INFO, "Saved context in " + (System.currentTimeMillis() - beginSaveContext) + " ms");
			long beginSaveSessionFile = System.currentTimeMillis();
			processor.saveStateFile(sessionFile);
			logger.log(INFO, "Saved session file to " + sessionFile + " in "
					+ (System.currentTimeMillis() - beginSaveSessionFile) + " ms");
		}

		String userMsg02 = "Current Date: March 13th 2020.";

		Consumer<LlamaCppInstructProcessor> process = (processor) -> {
			System.out.println(USER.name() + " :\n" + userMsg02);
			processor.write(USER, userMsg02);

			System.out.println(ASSISTANT.name() + " :\n");
			long begin = System.currentTimeMillis();
			try {
				processor.readMessage(System.out);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
			logger.log(INFO, "Generation took " + +(System.currentTimeMillis() - begin) + " ms");
		};

		// deterministic answer
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			long beginLoad = System.currentTimeMillis();
			processor.loadContextState(savedState);
			logger.log(INFO, "Loaded context from memory in " + (System.currentTimeMillis() - beginLoad) + " ms");

			process.accept(processor);
		}

		// with temperature
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(true); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			long beginLoad = System.currentTimeMillis();
			processor.loadStateFile(sessionFile);
			logger.log(INFO, "Loaded context from file in " + (System.currentTimeMillis() - beginLoad) + " ms");

			process.accept(processor);
		}

		logger.log(INFO, "Saved context state smoke tests PASSED");
	}

	/*
	 * STATIC UTILITIES
	 */
	/** CLI entry point. */
	public static void main(String[] args) throws Exception {
		new JjmlSmokeTests().main(Arrays.asList(args));
	}

	/**
	 * Writes the beginning of an integer buffer as a string. It has no side effect
	 * on the input buffer.
	 */
	static String logIntegers(IntBuffer in, int max, String separator) {
		StringBuilder sb = new StringBuilder();
		integers: for (int i = in.position(); i < in.limit(); i++) {
			if (i != in.position())
				sb.append(separator);
			if (i == max) {
				sb.append("...");
				break integers;
			}
			sb.append(Integer.toString(in.get(i)));
		}
		return sb.toString();
	}

	/*
	 * CLASSES
	 */
	static class LoadModelProgressCallback implements DoubleConsumer {
		private int lastPerctPrinted = -1;

		@Override
		public void accept(double progress) {
			char[] progressBar = new char[10];
			int perct = (int) (progress * 100);

			if (perct > lastPerctPrinted + 10 //
					|| lastPerctPrinted == -1 //
					|| progress == 1.0) {

				for (int i = 0; i < perct / 10; i++)
					progressBar[i] = '#';
				for (int i = perct / 10; i < 10; i++)
					progressBar[i] = '-';
				System.err.print("\r" + new String(progressBar));

				lastPerctPrinted = perct;
				if (progress == 1.0)
					System.out.print("\n");
			}
		}

	}
}
