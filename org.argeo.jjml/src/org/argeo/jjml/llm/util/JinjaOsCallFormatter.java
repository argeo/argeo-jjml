package org.argeo.jjml.llm.util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppInstructFormatter;

/**
 * Calls a Python script with the Jinja template as stdin, and the inputs as
 * key=value arguments (value can be either a string or a json structure).
 * <b>For testing and development purposes only<b>, typically when developing
 * new instruct formatters.
 */
public class JinjaOsCallFormatter implements LlamaCppInstructFormatter {
	public final static String ENV_JJML_JINJA_PYTHON_SCRIPT = "JJML_JINJA_PYTHON_SCRIPT";
	public final static String ENV_JJML_PYTHON_EXEC = "JJML_PYTHON_EXEC";

	private final String jinjaTemplate;
	private final Path jjmlJinjaPythonScript;
	private final Path pythonExec;

	public JinjaOsCallFormatter(String jinjaTemplate) {
		this(jinjaTemplate, getJinjaPythonScriptFromEnvironment());
	}

	public JinjaOsCallFormatter(String jinjaTemplate, Path jjmlJinjaPythonScript) {
		this.jinjaTemplate = jinjaTemplate;

		if (!Files.exists(jjmlJinjaPythonScript))
			throw new IllegalArgumentException("Python script " + jjmlJinjaPythonScript + " was not found.");
		this.jjmlJinjaPythonScript = jjmlJinjaPythonScript;

		String pythonExecStr = System.getenv(ENV_JJML_PYTHON_EXEC);
		if (pythonExecStr == null)
			pythonExecStr = "/usr/bin/python3";
		this.pythonExec = Paths.get(pythonExecStr);
		if (!Files.exists(pythonExec))
			throw new IllegalStateException("Python executable " + pythonExec + " was not found. It can be set via the "
					+ ENV_JJML_PYTHON_EXEC + " environment variable.");
		// TODO also support virtual environments
	}

	@Override
	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		String messagesStr = formatAsJson(messages);

		List<String> command = new ArrayList<>();
		command.add(pythonExec.toString());
		command.add(jjmlJinjaPythonScript.toString());
		command.add("messages=" + messagesStr);
//		if (!hasSystemPrompt(messages))
//			command.add("default_system_message=");

		ProcessBuilder processBuilder = new ProcessBuilder(command);
		processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);
		try {
			Process process = processBuilder.start();
			try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
				writer.write(jinjaTemplate);
				writer.flush();
			}

			byte[] outputBytes = process.getInputStream().readAllBytes();
			String outputString = new String(outputBytes, StandardCharsets.UTF_8);

			int exitCode = process.waitFor();
			// TODO deal with interrupt, corner cases, etc.
			if (exitCode != 0)
				throw new RuntimeException("Call to " + jjmlJinjaPythonScript + " failed with exit code " + exitCode);
			return outputString;

		} catch (IOException | InterruptedException e) {
			throw new RuntimeException("Cannot process Jinja template", e);
		}
	}

	/*
	 * STATIC UTILITIES
	 */
	private static Path getJinjaPythonScriptFromEnvironment() {
		String jjmlJinjaPythonScriptStr = System.getenv(ENV_JJML_JINJA_PYTHON_SCRIPT);
		if (jjmlJinjaPythonScriptStr == null)
			throw new IllegalArgumentException(
					"Environment variable " + ENV_JJML_JINJA_PYTHON_SCRIPT + " must be set.");
		return Paths.get(jjmlJinjaPythonScriptStr);
	}

	private static String formatAsJson(List<LlamaCppChatMessage> messages) {
		StringJoiner sjArr = new StringJoiner(",", "[", "]");
		for (LlamaCppChatMessage msg : messages) {
			StringJoiner sjObj = new StringJoiner(",", "{", "}");
			sjObj.add("\"role\":\"" + msg.getRole() + "\"");
			sjObj.add("\"content\":\"" + msg.getContent().replace("\n", "\\n") + "\"");
			sjArr.add(sjObj.toString());
		}
		return sjArr.toString();
	}

//	private static boolean hasSystemPrompt(List<LlamaCppChatMessage> messages) {
//		for (LlamaCppChatMessage message : messages) {
//			if (message.getRole().toUpperCase().equals(InstructRole.SYSTEM.name()))
//				return true;
//		}
//		return false;
//	}

	public static void main(String[] args) throws IOException {
		if (args.length == 0)
			throw new IllegalArgumentException(
					"Usage: " + JinjaOsCallFormatter.class.getSimpleName() + " <path to GGUF model>");
		Path jinjaTemplatePath = Paths.get(args[0]);
		if (!Files.exists(jinjaTemplatePath))
			throw new IllegalArgumentException("GGUF model " + jinjaTemplatePath + " was not found.");

		String jinjaTemplate = Files.readString(jinjaTemplatePath);
		JinjaOsCallFormatter instructFormatter = new JinjaOsCallFormatter(jinjaTemplate);
		String outputStr = instructFormatter.formatChatMessages(new LlamaCppChatMessage("user", "Hello world"));
		System.out.println(outputStr);
	}
}
