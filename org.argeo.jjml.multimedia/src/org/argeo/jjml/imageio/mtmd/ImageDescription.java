package org.argeo.jjml.imageio.mtmd;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.llm.util.SimpleModelDownload;
import org.argeo.jjml.mtmd.MtmdBackend;
import org.argeo.jjml.mtmd.MtmdBitmap;
import org.argeo.jjml.mtmd.MtmdContext;
import org.argeo.jjml.mtmd.MtmdImageBitmap;
import org.argeo.jjml.mtmd.MtmdNative;
import org.argeo.jjml.mtmd.MtmdProcessor;

public class ImageDescription {
	private final static String SYSTEM_PROMPT_MINISTRAL_THINK = """
			# HOW YOU SHOULD THINK AND ANSWER

			First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

			Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.
					""";

	public static void main(String[] args) throws Exception {
		if (args.length < 4)
			throw new IllegalArgumentException("Usage: " + ImageDescription.class.getSimpleName()
					+ "<path to model> <path to mmproj> <prompt> <path to image>");

		MtmdNative.ensureLibrariesLoaded();

		Path modelPath = Paths.get(args[0]);
		if (!Files.exists(modelPath))
			modelPath = SimpleModelDownload.getDefaultModelsBase().resolve(modelPath);
		if (!Files.exists(modelPath))
			throw new IllegalArgumentException("Cannot find model " + args[0]);

		Path mmprojPath = Paths.get(args[1]);
		if (!Files.exists(mmprojPath))
			mmprojPath = SimpleModelDownload.getDefaultModelsBase().resolve(mmprojPath);
		if (!Files.exists(mmprojPath))
			throw new IllegalArgumentException("Cannot find mmproj " + args[1]);

		String prompt = args[2];

		Path imagePath = Paths.get(args[3]);
		if (!Files.exists(imagePath))
			throw new IllegalArgumentException("Cannot find iamge " + args[3]);

//		Path modelPath = SimpleModelDownload.getDefaultModelsBase()
//				.resolve("ggml-org_Qwen2.5-VL-3B-Instruct-GGUF_Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf");
//		Path mmprojPath = SimpleModelDownload.getDefaultModelsBase()
//				.resolve("ggml-org_Qwen2.5-VL-3B-Instruct-GGUF_mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf");

//		Path modelPath = SimpleModelDownload.getDefaultModelsBase()
//				.resolve("ibm-granite_granite-vision-3.3-2b-GGUF_granite-vision-3.3-2b-Q8_0.gguf");
//		Path mmprojPath = SimpleModelDownload.getDefaultModelsBase()
//				.resolve("ibm-granite_granite-vision-3.3-2b-GGUF_mmproj-model-f16.gguf");

//		Path imagePath = Paths.get(System.getProperty("user.home"), //
////				"Pictures/test3.jpg" //
//				"Pictures/test2.png" //
//		);
		try (LlamaCppModel model = LlamaCppModel.load(modelPath);
				LlamaCppContext context = new LlamaCppContext(model,
						defaultContextParams().with(ContextParam.n_ctx, 4096)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
				MtmdContext mtmdContext = new MtmdContext(model, mmprojPath,
						Runtime.getRuntime().availableProcessors()); //
				InputStream imageIn = Files.newInputStream(imagePath); //
		) {
			MtmdProcessor processor = new MtmdProcessor(context, chain, mtmdContext);
//			LlamaCppChatMessage systemPrompt = new LlamaCppChatMessage(InstructRole.SYSTEM,
//					SYSTEM_PROMPT_MINISTRAL_THINK);
			LlamaCppChatMessage systemPrompt = null;
			String formatted = model.formatChatMessages(systemPrompt,
					new LlamaCppChatMessage(InstructRole.USER, //
					MtmdBackend.getDefaultMarker() + prompt));
			MtmdImageBitmap bitmap = ImageIoBitmap.load(imageIn);
			MtmdBitmap[] bitmaps = new MtmdBitmap[] { bitmap };

			long begin = System.currentTimeMillis();
			String response = processor.transcribe(formatted, bitmaps);
			System.out.println(response);
			System.out.println("Processing took " + (System.currentTimeMillis() - begin) + " ms");
		}
	}

}
