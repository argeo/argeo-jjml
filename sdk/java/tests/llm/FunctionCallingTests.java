package tests.llm;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_threads;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.IOException;
import java.util.AbstractMap;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.JavaMethodToolDef;
import org.argeo.jjml.llm.util.models.LlamaCppAgenticProcessor;

class FunctionCallingTests extends AbstractLlmTests {
	FunctionCallingTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testJavaIntroChat();
	}

	void testJavaIntroChat() throws IOException {
		try (//
				LlamaCppContext context = new LlamaCppContext(getModel(), defaultContextParams() //
						.with(n_ctx, 4096) //
						.with(n_batch, 1024) //
						.with(n_threads, getParallelism()) //
				); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppAgenticProcessor processor = new LlamaCppAgenticProcessor(context, chain);
			processor.setDebugPrompts(System.err);

			try {
				// java.lang.System
				JavaMethodToolDef getSystemProperty = new JavaMethodToolDef(
						System.class.getMethod("getProperty", String.class),
						"Gets the system property indicated by the specified key.",
						new AbstractMap.SimpleEntry<>("key", "the name of the system property."));
				processor.registerTool(getSystemProperty.getDefaultToolName(), getSystemProperty);

				JavaMethodToolDef getenv = new JavaMethodToolDef(System.class.getMethod("getenv", String.class),
						"Gets the value of the specified environment variable. An environment variable is a system-dependent external named value.",
						new AbstractMap.SimpleEntry<>("name", "the name of the environment variable"));
				processor.registerTool(getenv.getDefaultToolName(), getenv);

				// java.lang.Runtime
				JavaMethodToolDef freeMemory = new JavaMethodToolDef(Runtime.class.getMethod("freeMemory"),
						"Returns the amount of free memory in the Java Virtual Machine.");
				freeMemory.setDefaultContext(Runtime.getRuntime());
				processor.registerTool(freeMemory.getDefaultToolName(), freeMemory);

				JavaMethodToolDef totalMemory = new JavaMethodToolDef(Runtime.class.getMethod("totalMemory"),
						"Returns the total amount of memory in the Java virtual machine."
								+ " The value returned by this method may vary over time, depending on the host environment.");
				totalMemory.setDefaultContext(Runtime.getRuntime());
				processor.registerTool(totalMemory.getDefaultToolName(), totalMemory);

				JavaMethodToolDef maxMemory = new JavaMethodToolDef(Runtime.class.getMethod("maxMemory"),
						"Returns the maximum amount of memory that the Java virtual machine will attempt to use."
								+ " If there is no inherent limit then the value Long.MAX_VALUE will be returned.");
				maxMemory.setDefaultContext(Runtime.getRuntime());
				processor.registerTool(maxMemory.getDefaultToolName(), maxMemory);

			} catch (NoSuchMethodException | SecurityException e) {
				throw new RuntimeException("Cannot create tool based on Java method", e);
			}

			String systemMsg = "You are a helpful assistant, which answers as briefly as possible. " //
					+ "You are running within a Java Virtual Machine, and you have access to tools within this Java runtime. " //
			;
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "Using at least 10 tool calls and at most 30 tool calls,"//
					+ " summarize useful information about the running JVM," //
					+ " in the context of a test program testing your capabilities in calling tools. ";
			processor.write(USER, userMsg01);

			processor.readMessage(out);

			// make sure it can deal with a second message
			String userMsg02 = "Do it again! Try to be more relevant.";
			processor.write(USER, userMsg02);

			processor.readMessage(out);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 1)
			throw new IllegalArgumentException("Specify a model");
		String modelHint = args[0];
		ModelParams modelParams = defaultModelParams();
		try (LlamaCppModel model = AbstractLlmTests.createModel(modelParams, modelHint)) {
			new FunctionCallingTests(model).run();
		} finally {
			LlamaCppBackend.destroy();
		}
	}

}
