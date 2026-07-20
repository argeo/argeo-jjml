package org.argeo.jjml.llm.util.models;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;

import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.instruct.LlamaCppInstructFormatter;
import org.argeo.jjml.llm.instruct.LlamaCppInstructPart;
import org.argeo.jjml.llm.instruct.LlamaCppToolDefinition;
import org.argeo.jjml.llm.util.InstructRole;

public class LlamaCppAgenticProcessor extends LlamaCppInstructProcessor {

	private Map<String, LlamaCppToolDefinition> availableTools = new LinkedHashMap<>();

	private boolean callingTool = false;
	private String answerBeforeCallingTool = null;

	public LlamaCppAgenticProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppInstructFormatter instructFormatter) {
		super(context, samplerChain, instructFormatter);
	}

	public LlamaCppAgenticProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		super(context, samplerChain);
	}

	public void registerTool(String name, LlamaCppToolDefinition toolDefinition) {
		availableTools.put(name, toolDefinition);
		// StringJoiner tools = new StringJoiner(",", "[", "]");
		StringBuilder sb = new StringBuilder();
		toJson(sb, name, toolDefinition);
		System.out.println(sb);
	}

	@Override
	public void write(LlamaCppInstructPart message) {
		super.write(message);
		if (message.getRole().equals("system") && !availableTools.isEmpty()) {
			StringJoiner tools = new StringJoiner(",", "[", "]");
			for (String name : availableTools.keySet()) {
				StringBuilder sb = new StringBuilder();
				toJson(sb, name, availableTools.get(name));
				tools.add(sb);
			}
			writeFormatted("[AVAILABLE_TOOLS]" + tools + "[/AVAILABLE_TOOLS]");
		}
	}

	public String nextAnswer() {
		if (isGenerationCompleted(0))
			return null;
		IntBuffer output = getNextAnswerTokenBuffer();

		CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
		CompletableFuture<Boolean> allCompleted = readBatchAsync(new IntBuffer[] { output }, generationCompleted);
		allCompleted.join();

		output.flip();

		// int toolCallToken = 16000; // Ministral 310
		int toolCallToken = getVocabulary().tokenize("[TOOL_CALLS]").get(0); // Ministral 3
		// int toolArgToken = 32; // Ministral 3
		int toolArgToken = getVocabulary().tokenize("[ARGS]").get(0); // Ministral 3
		if (!callingTool) {
			// scan for tool call token
			scan: for (int i = output.position(); i < output.limit(); i++) {
				int value = output.get(i); // does not move the pointer
				if (value == toolCallToken) {
					callingTool = true;
					IntBuffer beforeCalling = output.duplicate();
					beforeCalling.limit(i);
					answerBeforeCallingTool = getVocabulary().deTokenize(beforeCalling);
					output.position(i);
					break scan;
				}
//				if (callingTool)
//					System.out.println(i + " : " + value);
			}
		}

		if (!callingTool) {
			String outputStr = getVocabulary().deTokenize(output);
			return outputStr;
		} else {
			ByteBuffer tokenBuffer = ByteBuffer.allocateDirect(1024 * Integer.BYTES);
			tokenBuffer.order(ByteOrder.nativeOrder());
			IntBuffer toolCallsBuffer = tokenBuffer.asIntBuffer();
			toolCallsBuffer.put(output); // remaining

			// decode
			CompletableFuture<Boolean>[] generationCompleted2 = newGenerationCompletableFutures();
			CompletableFuture<Boolean> allCompleted2 = readBatchAsync(new IntBuffer[] { toolCallsBuffer },
					generationCompleted2);
			allCompleted2.join();

			toolCallsBuffer.flip();

			List<IntBuffer> functionNames = new ArrayList<>();
			List<IntBuffer> functionArgs = new ArrayList<>();

//			int index = 0;
			scan: for (int i = toolCallsBuffer.position(); i < toolCallsBuffer.limit(); i++) {
				int value = toolCallsBuffer.get(i); // does not move the pointer
//				System.out.println(i + " : " + value);
				if (value == toolCallToken) {
					if (functionNames.size() != 0) {
						if (functionArgs.size() == functionNames.size()) {
							functionArgs.get(functionArgs.size() - 1).limit(i);
							// won't be called for the last one
							// but we assume the limit of the original buffer is ok
						} else { // no [ARGS] was found
							functionArgs.add(null);
							functionNames.get(functionNames.size() - 1).limit(i);
						}
					}
					IntBuffer buf = toolCallsBuffer.duplicate();
					buf.position(i + 1);
					functionNames.add(buf);
				} else if (value == toolArgToken) {
					functionNames.get(functionNames.size() - 1).limit(i);

					IntBuffer buf = toolCallsBuffer.duplicate();
					buf.position(i + 1);
					functionArgs.add(buf);
				}
			}

			if (functionNames.size() == (functionArgs.size() + 1))
				functionArgs.add(null); // last one was without args

			if (functionNames.size() != functionArgs.size())
				throw new IllegalStateException("Not as many functions (" + functionNames.size() + ") as arguments ("
						+ functionArgs.size() + ")");
			List<Map<String, Object>> results = new ArrayList<>();
			for (int i = 0; i < functionNames.size(); i++) {
				String functionName = getVocabulary().deTokenize(functionNames.get(i));
				LlamaCppToolDefinition toolDefinition = availableTools.get(functionName);
				if (toolDefinition == null)
					throw new IllegalArgumentException("Tool '" + functionName + "' was not found");

				Map<String, Object> args;
				IntBuffer fArg = functionArgs.get(i);
				if (fArg != null) {
					String functionArg = getVocabulary().deTokenize(fArg);
					args = (Map<String, Object>) SimpleJsonParser.parse(functionArg);
				} else {
					args = null;// empty
				}

				// EXECUTION
				if (debugPrompts != null)
					debugPrompts.println("\n#! Tool call " + i + " - " + functionName + " with args " + args + " ...");
				CompletionStage<Map<String, Object>> executed = toolDefinition.execute(null, args);
				//

				try {
					Map<String, Object> executedRes = executed.toCompletableFuture().get();
					results.add(executedRes);

					StringBuilder sb = new StringBuilder();
					toJson(sb, executedRes);
					LlamaCppInstructPart toolPart = new LlamaCppChatMessage(InstructRole.TOOL, sb.toString());
					write(toolPart);
				} catch (InterruptedException | ExecutionException e) {
					throw new RuntimeException("Cannot execute function call " + i + " - " + functionName, e);
				}
			}

			callingTool = false;
			return answerBeforeCallingTool;
		}
	}

	protected void toJson(StringBuilder sb, Map<String, Object> result) {
		StringJoiner sj = new StringJoiner(",", "{", "}");
		values: for (String key : result.keySet()) {
			Object value = result.get(key);
			if (value == null) {
				sj.add("\"" + key + "\":null");
				continue values;
			} else if (value instanceof Number || value instanceof Boolean) {
				sj.add("\"" + key + "\":" + value);
				continue values;
			}
			String str = value.toString();
			if ("false".equals(str.toLowerCase()) || "true".equals(str.toLowerCase())) {
				Boolean b = Boolean.parseBoolean(str.toLowerCase());
				sj.add("\"" + key + "\":" + b);
				continue values;
			}
			try {
				Long l = Long.parseLong(str);
				sj.add("\"" + key + "\":" + l);
				continue values;
			} catch (NumberFormatException e) {
				// silent
			}
			try {
				Double d = Double.parseDouble(str);
				sj.add("\"" + key + "\":" + d);
				continue values;
			} catch (NumberFormatException e) {
				// silent
			}

			// falback as string
			sj.add("\"" + key + "\":\"" + str + "\"");
		}
		sb.append(sj);
	}

	protected void toJson(StringBuilder sb, String functionName, LlamaCppToolDefinition toolDefinition) {
		// FIXME sanitize inputs

		sb.append("{\"type\":\"function\",");

		sb.append("\"function\":{");

		sb.append("\"name\":\"" + functionName + "\",");
		sb.append("\"description\":\"" + toolDefinition.getDescription() + "\",");
		{
			sb.append("\"parameters\":{");
			sb.append("\"type\":\"object\",");

			// sb.append("\"properties\":{");
			StringJoiner props = new StringJoiner(",", "\"properties\":{", "}");
			StringJoiner required = new StringJoiner(",", "\"required\":[", "]");
			for (String paramName : toolDefinition.getParameters().keySet()) {
				LlamaCppToolDefinition.Param param = toolDefinition.getParameters().get(paramName);
				StringBuilder prop = new StringBuilder();
				prop.append("\"" + paramName + "\":{");
				prop.append("\"type\":\"" + param.getType().get() + "\",");
				prop.append("\"description\":\"" + param.getDescription() + "\"");
				prop.append("}");
				props.add(prop);

				required.add("\"" + paramName + "\"");
			}
			// sb.append("}");
			sb.append(props.toString());
			sb.append(",");
			sb.append(required.toString());

			sb.append("}");
		}
		sb.append("}");

		sb.append("}");
	}

}
