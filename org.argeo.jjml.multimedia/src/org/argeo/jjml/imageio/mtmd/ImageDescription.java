package org.argeo.jjml.imageio.mtmd;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.argeo.jjml.llm.LLamaCppNativeChatFormatter;
import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.mtmd.MtmdBackend;
import org.argeo.jjml.mtmd.MtmdBitmap;
import org.argeo.jjml.mtmd.MtmdContext;
import org.argeo.jjml.mtmd.MtmdImageBitmap;
import org.argeo.jjml.mtmd.MtmdNative;
import org.argeo.jjml.mtmd.MtmdProcessor;

public class ImageDescription {
	public static void main(String[] args) throws Exception {
		if (args.length < 4)
			throw new IllegalArgumentException("Usage: " + ImageDescription.class.getSimpleName()
					+ "<path to model> <path to mmproj> <prompt> <path to image>");

		MtmdNative.ensureLibrariesLoaded();

		Path modelPath = Paths.get(args[0]);
		if (!Files.exists(modelPath))
			throw new IllegalArgumentException("Cannot find model " + args[0]);

		Path mmprojPath = Paths.get(args[1]);
		if (!Files.exists(mmprojPath))
			throw new IllegalArgumentException("Cannot find mmproj " + args[1]);

		String prompt = args[2];

		Path imagePath = Paths.get(args[3]);
		if (!Files.exists(imagePath))
			throw new IllegalArgumentException("Cannot find image " + args[3]);

		try (LlamaCppModel model = LlamaCppModel.load(modelPath);
				LlamaCppContext context = new LlamaCppContext(model,
						defaultContextParams().with(ContextParam.n_ctx, 4096)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
				MtmdContext mtmdContext = new MtmdContext(model, mmprojPath,
						Runtime.getRuntime().availableProcessors()); //
				InputStream imageIn = Files.newInputStream(imagePath); //
		) {
			MtmdProcessor processor = new MtmdProcessor(context, chain, mtmdContext);
			LlamaCppChatMessage systemPrompt = null;
			LlamaCppChatMessage userPrompt = new LlamaCppChatMessage(InstructRole.USER, //
					MtmdBackend.getDefaultMarker() + prompt);
			String formatted = new LLamaCppNativeChatFormatter(model.getMetadataChatTemplate())
					.formatChatMessages(systemPrompt, userPrompt);
			MtmdImageBitmap bitmap = ImageIoBitmap.load(imageIn);
			MtmdBitmap[] bitmaps = new MtmdBitmap[] { bitmap };

			long begin = System.currentTimeMillis();
			String response = processor.transcribe(formatted, bitmaps);
			System.out.println(response);
			System.out.println("\nProcessing took " + (System.currentTimeMillis() - begin) + " ms");
		}
	}

}
