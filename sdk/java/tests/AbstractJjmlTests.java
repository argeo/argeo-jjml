package tests;

import static java.lang.System.Logger.Level.DEBUG;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.Logger.Level.WARNING;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.lang.System.Logger;

public abstract class AbstractJjmlTests implements Runnable {
	public static boolean allPassed = true;

	protected final Logger logger = System.getLogger(getClass().getName());
	protected final PrintStream out = logger.isLoggable(DEBUG) ? System.out
			: new PrintStream(OutputStream.nullOutputStream());

	protected abstract void all() throws IOException, InterruptedException;

	@Override
	public void run() {
		try {
			all();
			logger.log(INFO, "PASSED - " + getClass().getSimpleName());
		} catch (IOException e) {
			allPassed = false;
			throw new UncheckedIOException(" ERROR  - " + getClass().getSimpleName() + " - " + e.getMessage(), e);
		} catch (Exception | AssertionError e) {
			allPassed = false;
			logger.log(WARNING, "FAILED - " + getClass().getSimpleName() + " - " + e.getMessage());
			if (logger.isLoggable(INFO))
				e.printStackTrace();
		}
	}

}
