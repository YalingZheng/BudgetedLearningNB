package dk.app.core;

/**
 * an interface that represents something that can interpret and execute a workflow(job) definition.
 * @author kdeng
 *
 */
public interface IntfWFDExecutor {
	
	public enum RunMode{ASYNC, SYNC};
	
	/**
	 * intepret the wfd.
	 * @param wfd
	 * @return true if wfd is understood by this executor, false otherwise.
	 */
	public boolean interpret(IntfWFD wfd);
	
	
	/**
	 * 
	 * @param e
	 * @return true if it believes that it's compatible with the provided execution environment,
	 * false otherwise.
	 */
	public boolean isCompatible(IntfEnv e);
	
	
	
	/**
	 * 
	 * @return the current state of the executor if possible. 
	 * state may not be accessible if the "run" mode of this executor is synced, however 
	 * this can be "overridden" by the "run" mode of the engine. 
	 */
	public StateKind getState();
	
	
	/**
	 * @return the result of this executor.
	 */
	public Object getResult();
	
	
	/**
	 * force the run mode to be either sync or async, 
	 * 
	 * @return the effective run mode. 
	 */
	public IntfWFDExecutor.RunMode setRunMode(IntfWFDExecutor.RunMode rm);
	

	public IntfWFDExecutor.RunMode getRunMode();
		
	
	public void setEnv(IntfEnv env);
	
	public IntfEnv getEnv();
	
	/**
	 * should be called after interpret and setEnv.
	 *
	 */
	public void run();
	
	
	
	/**
	 * equiv to first set the current env and the wfd and then run. 
	 * @param env
	 */
	public void run(IntfEnv env, IntfWFD wfd);
	
//	
//	public boolean pause();
//	
//	public void resume();
//	
//	public boolean terminate();
	
	
}
