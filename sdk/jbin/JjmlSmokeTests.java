
//!/usr/bin/env -S java -ea -cp /usr/share/java/org.argeo.jjml.jar
import static java.lang.System.Logger.Level.DEBUG;
import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.Logger.Level.WARNING;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.lang.System.Logger;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
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
import org.argeo.jjml.llm.util.SimpleProgressCallback;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
class JjmlSmokeTests {
	private final static Logger logger = System.getLogger(JjmlSmokeTests.class.getName());

	private int parallelism = Runtime.getRuntime().availableProcessors();

	@SuppressWarnings("deprecation")
	public void main(List<String> args) throws Exception, AssertionError {
		try {
			if (!getClass().desiredAssertionStatus()) {
				logger.log(ERROR, "Assertions must be enabled. Please call Java with the -ea option.");
				return;
			}

			long begin = System.currentTimeMillis();

			// even without a model we can check whether native libraries are loading
			assert ((BooleanSupplier) () -> {
				LlamaCppNative.ensureLibrariesLoaded();
				return true;
			}).getAsBoolean();
			logger.log(INFO, "Native libraries loaded properly.");

			if (args.isEmpty()) {
				logger.log(WARNING, "No model was specified, only loading the native libraries was tested");
				return;
			}

			String arg0 = args.get(0);
			Path modelPath = Paths.get(arg0);
			if (!Files.exists(modelPath))
				try {
					modelPath = new HfModelDownload().getOrDownloadModel(arg0, new SimpleProgressCallback());
				} catch (Exception e) {
					logger.log(WARNING, "Using legacy model download\n(reason: " + e + ")");
					// Legacy downloads for unattended tests
					modelPath = new SimpleModelDownload().getOrDownloadModel(arg0, new SimpleProgressCallback());
				}
			if (!Files.exists(modelPath))
				throw new IllegalArgumentException("Could not find GGUF model " + modelPath);

			ModelParams modelParams = defaultModelParams();
			logger.log(INFO, "Loading model " + modelPath + " ...");
			Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams, new SimpleProgressCallback(),
					null);
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
//				assertEmbeddings(model);
//				assertBatch(model);
//				assertJavaSampler(model);
				assertChat(model);
				assertSavedContextState(model);
			}
			logger.log(INFO, "Smoke tests passed in " + (System.currentTimeMillis() - begin) / 1000 + " s with model "
					+ modelPath.getFileName());
		} catch (Exception | AssertionError e) {
			logger.log(ERROR, "Smoke tests failed", e);
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
						.with(n_ctx, 2048) //
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
				.with(n_ctx, 2048) //
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

	/** Print required arguments. */
	static void printUsage() {
		System.err.println("Usage: java " + JjmlSmokeTests.class.getName() + //
				".java path/to/model.gguf | hf_repo/model[:quantization]\n" + //
				"e.g. java " + JjmlSmokeTests.class.getName() + ".java allenai/OLMo-2-0425-1B-Instruct-GGUF");
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
}

//
// LEGACY DOWNLOAD MECHANISMS
//
/**
 * Downloads a GGUF model (by defaults from HuggingFace) with the same naming
 * conventions as llama-cli. This is meant to be used for prototyping, not as a
 * full-fledged models management solution.
 */
class HfModelDownload {
	private int bufferSize = 1024 * 4096;

	private final Path modelsBase;

	public HfModelDownload(Path modelsBase) {
		this.modelsBase = modelsBase;
	}

	public HfModelDownload() {
		this(getDefaultModelsBase());
	}

	public URL getRemoteUrl(String hfRepo, String quantization) {
		URI uri = getRemoteUri(hfRepo, quantization);
		try {
			return uri.toURL();
		} catch (MalformedURLException e) {
			throw new IllegalArgumentException("Malformed download URL: " + uri, e);
		}

	}

	/** To be overridden in order to use something else than HuggingFace. */
	public URI getRemoteUri(String hfRepo, String quantization) {
		return URI.create("https://huggingface.co/" + hfRepo + "/resolve/main/"
				+ getRemoteFileName(hfRepo, quantization) + "?download=true");
	}

	public String getRemoteFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		return fileName;
	}

	public String getLocalFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
//		String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
		String localFileName = fileName;
		return localFileName;
	}

	public Path getLocalHfRepoBaseDir(String hfRepo) {
		return modelsBase.resolve("models--" + hfRepo.replace("/", "--"));
	}

	public Path getLocalFile(String hfRepo, String quantization) throws IOException {
		Path baseDir = getLocalHfRepoBaseDir(hfRepo);
		Path refsFile = baseDir.resolve("refs").resolve("main");
		if (!Files.exists(refsFile))
			return null;
		String ref = Files.readString(refsFile).strip();
		Path modelsDir = baseDir.resolve("snapshots").resolve(ref);
		return modelsDir.resolve(getLocalFileName(hfRepo, quantization));
	}

	public Path getOrDownloadModel(String hfRepoArg, DoubleConsumer progressCallback) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getOrDownloadModel(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1], progressCallback);
		} else {
			return getOrDownloadModel(hfRepoArg, "Q4_K_M", progressCallback);
		}
	}

	public Path getOrDownloadModel(String hfRepo, String quantization, DoubleConsumer progressCallback)
			throws IOException {
		Path localFile = getLocalFile(hfRepo, quantization);

		String currentEtag = null;
		if (localFile != null && Files.exists(localFile)) {
			try {
				byte[] buf = null; // (byte[]) Files.readAttributes(localFile, "user:etag").getOrDefault("etag",
									// null);
				if (buf != null)
					currentEtag = new String(buf, StandardCharsets.US_ASCII);
			} catch (Exception e) {
//				logger.log(DEBUG, "Cannot read attribute from " + localFile, e);
			}
			if (currentEtag == null) {
				return localFile;
//				throw new IllegalStateException("File " + localFile + " already exist, remove it first");
			}
		} else {
			throw new UnsupportedOperationException(
					"Downloading files is currently not supported. Use llama.cpp tools to download models first.");
		}

		Files.createDirectories(getLocalHfRepoBaseDir(hfRepo));

		URL url = getRemoteUrl(hfRepo, quantization);
		HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
		if (currentEtag != null)
			urlConnection.setRequestProperty("If-None-Match", currentEtag);
		urlConnection.connect();
		// TODO somehow show download progress
		try (InputStream in = urlConnection.getInputStream()) {
			int responseCode = urlConnection.getResponseCode();
			double contentLength = urlConnection.getContentLengthLong();
			String etag = urlConnection.getHeaderField("etag");
			if (responseCode == HttpURLConnection.HTTP_NOT_MODIFIED || etag.equals(currentEtag)) {
//				logger.log(INFO, "Use unmodified " + hfRepo + ":" + quantization + " file with etag=" + currentEtag);
				return localFile;
			}
//			logger.log(INFO, "Starting download of " + url + " to " + localFile);
//			logger.log(DEBUG, "Effective URL " + urlConnection.getURL());

//			long begin = System.currentTimeMillis();
			try (OutputStream out = Files.newOutputStream(localFile, CREATE, TRUNCATE_EXISTING)) {
				byte[] buf = new byte[bufferSize];
				double downloaded = 0;
				int len;
				while ((len = in.read(buf)) != -1) {
					out.write(buf, 0, len);
					downloaded = downloaded + len;
					if (progressCallback != null)
						progressCallback.accept(downloaded / contentLength);
				}
			}
			// Files.copy(in, localFile, StandardCopyOption.REPLACE_EXISTING);
			if (etag != null) {
//				Files.setAttribute(localFile, "user:etag", StandardCharsets.US_ASCII.encode(etag));
			}
		}
		return localFile;
	}

	/*
	 * STATIC
	 */
	/** The default path where GGUF files are downloaded and searched for. */
	public static Path getDefaultModelsBase() {
		Path defaultModelsBase;
		String os = System.getProperty("os.name").toLowerCase();
		if (os.contains("win")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");
		} else if (os.contains("mac") || os.contains("darwin")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "Library", "Caches", "llama.cpp");
		} else { // Linux / Unix
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "huggingface", "hub");
		}
		return defaultModelsBase;
	}
}

/**
 * Downloads a GGUF model (by defaults from HuggingFace) with the same legacy
 * naming conventions as llama-cli. This is meant to be used for prototyping,
 * not as a full-fledged models management solution.
 * 
 * @deprecated Still uses the legacy llama.cpp download format.
 * @see HfModelDownload
 */
@Deprecated
class SimpleModelDownload {
	private int bufferSize = 1024 * 4096;

	private final Path modelsBase;

	public SimpleModelDownload(Path modelsBase) {
		this.modelsBase = modelsBase;
	}

	public SimpleModelDownload() {
		this(getDefaultModelsBase());
	}

	public URL getRemoteUrl(String hfRepo, String quantization) {
		URI uri = getRemoteUri(hfRepo, quantization);
		try {
			return uri.toURL();
		} catch (MalformedURLException e) {
			throw new IllegalArgumentException("Malformed download URL: " + uri, e);
		}

	}

	/** To be overridden in order to use something else than HuggingFace. */
	public URI getRemoteUri(String hfRepo, String quantization) {
		return URI.create("https://huggingface.co/" + hfRepo + "/resolve/main/"
				+ getRemoteFileName(hfRepo, quantization) + "?download=true");
	}

	public String getRemoteFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		return fileName;
	}

	public String getLocalFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + ".gguf");
		String localFileName = hfRepo.replace("/", "_") + "_" + fileName;
		return localFileName;
	}

	public Path getLocalFile(String hfRepo, String quantization) {
		return modelsBase.resolve(getLocalFileName(hfRepo, quantization));
	}

	public Path getOrDownloadModel(String hfRepoArg, DoubleConsumer progressCallback) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getOrDownloadModel(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1], progressCallback);
		} else {
			return getOrDownloadModel(hfRepoArg, "Q4_K_M", progressCallback);
		}
	}

	public Path getOrDownloadModel(String hfRepo, String quantization, DoubleConsumer progressCallback)
			throws IOException {
		Path localFile = getLocalFile(hfRepo, quantization);

		String currentEtag = null;
		if (Files.exists(localFile)) {
			try {
				byte[] buf = null; // (byte[]) Files.readAttributes(localFile, "user:etag").getOrDefault("etag",
									// null);
				if (buf != null)
					currentEtag = new String(buf, StandardCharsets.US_ASCII);
			} catch (Exception e) {
//				logger.log(DEBUG, "Cannot read attribute from " + localFile, e);
			}
			if (currentEtag == null) {
				return localFile;
//				throw new IllegalStateException("File " + localFile + " already exist, remove it first");
			}
		}
		Files.createDirectories(localFile.getParent());

		URL url = getRemoteUrl(hfRepo, quantization);
		HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
		if (currentEtag != null)
			urlConnection.setRequestProperty("If-None-Match", currentEtag);
		urlConnection.connect();
		// TODO somehow show download progress
		try (InputStream in = urlConnection.getInputStream()) {
			int responseCode = urlConnection.getResponseCode();
			double contentLength = urlConnection.getContentLengthLong();
			String etag = urlConnection.getHeaderField("etag");
			if (responseCode == HttpURLConnection.HTTP_NOT_MODIFIED || etag.equals(currentEtag)) {
//				logger.log(INFO, "Use unmodified " + hfRepo + ":" + quantization + " file with etag=" + currentEtag);
				return localFile;
			}
//			logger.log(INFO, "Starting download of " + url + " to " + localFile);
//			logger.log(DEBUG, "Effective URL " + urlConnection.getURL());

//			long begin = System.currentTimeMillis();
			try (OutputStream out = Files.newOutputStream(localFile, CREATE, TRUNCATE_EXISTING)) {
				byte[] buf = new byte[bufferSize];
				double downloaded = 0;
				int len;
				while ((len = in.read(buf)) != -1) {
					out.write(buf, 0, len);
					downloaded = downloaded + len;
					if (progressCallback != null)
						progressCallback.accept(downloaded / contentLength);
				}
			}
			// Files.copy(in, localFile, StandardCopyOption.REPLACE_EXISTING);
			if (etag != null) {
//				Files.setAttribute(localFile, "user:etag", StandardCharsets.US_ASCII.encode(etag));
			}
		}
		return localFile;
	}

	/*
	 * STATIC
	 */
	/** The default path where GGUF files are downloaded and searched for. */
	public static Path getDefaultModelsBase() {
		Path defaultModelsBase;
		String os = System.getProperty("os.name").toLowerCase();
		if (os.contains("win")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "AppData", "Local", "llama.cpp");
		} else if (os.contains("mac") || os.contains("darwin")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), "Library", "Caches", "llama.cpp");
		} else { // Linux / Unix
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "llama.cpp");
		}
		return defaultModelsBase;
	}
}
