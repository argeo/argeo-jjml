package org.argeo.jjml.llama.util;

import static org.argeo.jjml.llama.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llama.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llama.params.ContextParam.n_ctx;
import static org.argeo.jjml.llama.params.ModelParam.n_gpu_layers;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CompletionStage;
import java.util.stream.Collectors;

import org.argeo.jjml.llama.LlamaCppContext;
import org.argeo.jjml.llama.LlamaCppModel;
import org.argeo.jjml.llama.LlamaCppSamplerChain;
import org.argeo.jjml.llama.params.ContextParam;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

public class SimpleChatHttpServer extends SimpleChat implements Runnable {
	private final InetSocketAddress address;

	private volatile boolean running;
	private HttpServer httpServer;

	public SimpleChatHttpServer(InetSocketAddress address, String systemPrompt, LlamaCppContext context,
			LlamaCppSamplerChain samplerChain) {
		super(systemPrompt, context);
		this.address = address;
	}

	public SimpleChatHttpServer(InetSocketAddress address, String systemPrompt, LlamaCppContext context) {
		this(address, systemPrompt, context, null);
	}

	@Override
	public void run() {
		try {
			httpServer = HttpServer.create(address, 0);
			httpServer.createContext("/", (exchange) -> {
				exchange.sendResponseHeaders(200, 0);
				try (BufferedReader reader = new BufferedReader(
						new InputStreamReader(exchange.getRequestBody(), StandardCharsets.UTF_8))) {
					String input = reader.lines().collect(Collectors.joining("\\\n"));
					apply(input, (str) -> {
						try {
							OutputStream out = exchange.getResponseBody();
							out.write(str.getBytes(StandardCharsets.UTF_8));
							out.flush();
						} catch (IOException e) {
							throw new UncheckedIOException(e);
						}
					}).thenAccept((v) -> {
						exchange.close();
					}).toCompletableFuture().join();

				}
			});
			Runtime.getRuntime().addShutdownHook(new Thread(() -> stop(), "Shutdown HTTP server"));
			httpServer.start();
			running = true;

			// wait
			while (running)
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					stop();
				}
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	public void stop() {
		cancelCurrentRead();
		httpServer.stop(60);
		httpServer = null;
		running = false;
	}

	public static void main(String... args) throws Exception {
		if (args.length == 0) {
			System.err.println("A model must be specified");
			printUsage(System.err);
			System.exit(1);
		}
		String firstArg = args[0];
		String[] modelArr = firstArg.split("@");
		Path modelPath = Paths.get(modelArr[0]);

		String hostname = null;
		int port = 0;
		if (modelArr.length > 1) {
			String target = modelArr[1];
			try {
				port = Integer.parseInt(target);
			} catch (NumberFormatException e) {
				String[] arr = target.split(":");
				hostname = arr[0];
				port = Integer.parseInt(arr[1]);
			}
		}

		String systemPrompt = "You are a helpful assistant.";
		int gpuLayers = 0;
		int contextSize = 4096;
		if (args.length > 1)
			systemPrompt = args[1];
		if (args.length > 2)
			gpuLayers = Integer.parseInt(args[2]);
		if (args.length > 3)
			contextSize = Integer.parseInt(args[3]);

		InetSocketAddress address = hostname == null ? new InetSocketAddress(port)
				: new InetSocketAddress(hostname, port);

		try (LlamaCppModel model = LlamaCppModel.load(modelPath, defaultModelParams() //
				.with(n_gpu_layers, gpuLayers)); //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, contextSize) //
						.with(ContextParam.n_batch, 32) //
				); //
		) {
			SimpleChatHttpServer chatServer = new SimpleChatHttpServer(address, systemPrompt, context);
			chatServer.run();
		}
	}

	private static void printUsage(PrintStream out) {
		// TODO make it clearer and specify system properties
		out.print("java " //
				+ "-Djava.library.path=/path/to/jjml_dll:/path/to/llama_dlls" //
				+ " -cp /path/to/org.argeo.jjml.jar " + SimpleChat.class.getName() //
				+ " <path/to/model.gguf> [<system prompt>] [<n_gpu_layers>] [<n_ctx>]");
	}

}
