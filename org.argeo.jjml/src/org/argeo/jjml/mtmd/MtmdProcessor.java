package org.argeo.jjml.mtmd;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.IntBuffer;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppSamplerChain;

public class MtmdProcessor {
	private final LlamaCppContext context;
	private final LlamaCppSamplerChain chain;
	private final MtmdContext mtmdContext;

	public MtmdProcessor(LlamaCppContext context, LlamaCppSamplerChain chain, MtmdContext mtmdContext) {
		this.context = context;
		this.chain = chain;
		this.mtmdContext = mtmdContext;
	}

	private static native int[] doSingleTurn(long contextPointer, long samplerPointer, long mtmdContextPointer,
			byte[] prompt, MtmdBitmap[] bitmaps);

	public String transcribe(String prompt, MtmdBitmap[] bitmaps) {
		int[] tokens = doSingleTurn(context.getAsLong(), chain.getAsLong(), mtmdContext.getAsLong(),
				prompt.getBytes(UTF_8), bitmaps);
		String response = context.getModel().getVocabulary().deTokenize(IntBuffer.wrap(tokens));
		return response;
	}
}