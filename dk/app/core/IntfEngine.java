package dk.app.core;

/**
 * provides another level of abstraction  which endows an environment to executor and controls its running.
 * Engine is not absolutely necessary for the framework, but it is convenient to have:
 * async "run"  inherent sequential executor;
 * provides info about the executor;
 * initialize environment and provide it to the executor;
 * start pause, resume, terminate the controlled executor.
 * 
 * @author kdeng
 *
 */
public interface IntfEngine {
	public void setEnv(IntfEnv env);
	public void setExecutor(IntfWFDExecutor e);
	public void setExecutorInfo(Object o);
	
	/**
	 * start the execution
	 *
	 */
	public void start(boolean sync);
	
	/**
	 * onestep shortcut to setenv/setexecutor/start
	 * @param e
	 * @param env
	 */
	public void start(IntfWFDExecutor e, IntfEnv env, Object info, boolean sync);
	public StateKind state();
	public void pause();
	public void resume();
	public void terminate();
	
	public IntfWFDExecutor getExecutor();
	public IntfEnv getEnv();
	public Object getExecutorInfo();
	// restart???
}
