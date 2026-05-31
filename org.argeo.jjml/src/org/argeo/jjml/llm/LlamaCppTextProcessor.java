package org.argeo.jjml.llm;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Collections;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;

/** A processor based on text rather than tokens. */
public class LlamaCppTextProcessor extends LlamaCppBatchProcessor {
	private final LlamaCppVocabulary vocabulary;

	public LlamaCppTextProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		this(context, samplerChain, null, Collections.singleton(0));
	}

	public LlamaCppTextProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppNativeSampler validatingSampler, Set<Integer> sequenceIds) {
		super(context, samplerChain, validatingSampler, sequenceIds);
		this.vocabulary = context.getModel().getVocabulary();
	}

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt) {
		return processBatch(systemPrompt);
	}

	public String processBatch(String prompt) {
		return processBatch(prompt, null, null);
	}

	public String processBatch(String prompt, String[] parameters, String postPrompt) {
		IntBuffer promptTokens = vocabulary.tokenize(prompt);
		assert promptTokens.position() == 0;
		int tokenCount = promptTokens.limit();
		int[] promptArr = promptTokens.array();

		int outputMax = getContext().getBatchSize();

		// TODO check whether it makes sense (pattern was taken from llama.cpp code)
		int requiredContextSize = tokenCount + outputMax * getParallelCount() * 10;

		int contextSize = getContext().getContextSize();
		if (getContext().getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");

		boolean direct = false;
		// direct buffer area
		IntBuffer buf;
		if (direct) {
			ByteBuffer directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
			buf = directBuf.asIntBuffer();
		} else {
			buf = IntBuffer.allocate(requiredContextSize);
		}

		int batchSize = getContext().getBatchSize();

		boolean tokenList = true;

		if (tokenList) {
			int batchCount = tokenCount / batchSize;
			if (tokenCount % batchSize != 0)
				batchCount = batchCount + 1;
			for (int i = 0; i < batchCount; i++) {
				IntBuffer input = buf.slice();
				boolean lastLogits;
				if (i == batchCount - 1) {
					input.limit(tokenCount % batchSize == 0 ? batchSize : tokenCount % batchSize);
					lastLogits = parameters == null;
				} else {
					input.limit(batchSize);
					lastLogits = false;
				}
				buf.position(buf.position() + input.limit());

				// copy data
				input.put(promptArr, i * batchSize, input.limit());
				input.flip();

				writeBatch(new IntBuffer[] { input }, lastLogits);
			}

			if (parameters != null) {
				if (parameters.length != getParallelCount())
					throw new IllegalArgumentException("Parameters count different from sequence count");

				IntBuffer[] inputs = new IntBuffer[getParallelCount()];
				for (int i = 0; i < getParallelCount(); i++) {
					IntBuffer parametersTokens = vocabulary.tokenize(parameters[i]);
					if (parametersTokens.remaining() * getParallelCount() > batchSize)// TODO be more precise / robust
						throw new IllegalArgumentException("Parameter '" + parameters[i] + "' is too long.");
					inputs[i] = buf.slice();
					inputs[i].limit(parametersTokens.remaining());
					buf.position(buf.position() + inputs[i].limit());

					// copy data
					inputs[i].put(parametersTokens.array(), 0, inputs[i].limit());
					inputs[i].flip();
				}
				writeBatch(inputs, postPrompt == null);
			}

			if (postPrompt != null) {
				IntBuffer postPromptTokens = vocabulary.tokenize(postPrompt);
				if (postPromptTokens.remaining() > batchSize)// TODO be more precise / robust
					throw new IllegalArgumentException("Post prompt '" + postPrompt + "' is too long.");
				IntBuffer input = buf.slice();
				input.limit(postPromptTokens.remaining());
				buf.position(buf.position() + input.limit());

				// copy data
				input.put(postPromptTokens.array(), 0, input.limit());
				input.flip();

				writeBatch(new IntBuffer[] { input }, true);
			}
		} else {
			IntBuffer input = buf.slice();
			vocabulary.tokenize(prompt, input, true, true);
			buf.position(input.position());

			input.flip();
			writeBatch(new IntBuffer[] { input }, true);
		}

		StringBuffer[] outputStrings = new StringBuffer[getParallelCount()];
		for (int i = 0; i < outputStrings.length; i++)
			outputStrings[i] = new StringBuffer();

		boolean reading = true;
		reads: while (reading) {
			IntBuffer[] outputs = new IntBuffer[getParallelCount()];
			outputs: for (int i = 0; i < getParallelCount(); i++) {
				if (isGenerationCompleted(i)) {
					outputs[i] = null;
					continue outputs;
				}
				IntBuffer output = buf.slice();
				output.limit(outputMax);
				outputs[i] = output;
				buf.position(buf.position() + output.limit());
			}

//			long begin = System.nanoTime();

			CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
			CompletableFuture<Boolean> allCompleted = readBatchAsync(outputs, generationCompleted);
			allCompleted.join();

//			long end = System.nanoTime();
//			System.out.println("Read  batch in " + (end - begin) / 1000000 + " ms.");

			int sequencesLeft = 0;
			for (int i = 0; i < outputs.length; i++) {
				IntBuffer output = outputs[i];
				if (output != null) {
					output.flip();
					String outputStr = vocabulary.deTokenize(output);
					outputStrings[i].append(outputStr);
				}

				if (!isGenerationCompleted(i)) {
					sequencesLeft++;
				} else {

				}
			}

			if (sequencesLeft == 0)
				break reads;

			System.out.println(sequencesLeft + " sequences left");

			if (buf.position() + sequencesLeft * outputMax > buf.capacity()) {
				System.err.println("Main buffer will be full, aborting...");
				break reads;
			}

			// TODO check context size and break the loop
			// TODO timeout?
		}
		StringJoiner res = new StringJoiner(
				"\n\n\n---------------------------------------------------------------\n\n\n");
		for (int i = 0; i < outputStrings.length; i++)
			res.add(outputStrings[i]);
		return res.toString();
	}
}
