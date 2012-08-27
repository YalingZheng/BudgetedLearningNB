package dk.blfw.exp;

import dk.app.core.IntfEnv;
import dk.app.core.IntfWFD;
import dk.app.core.IntfWFDExecutor;
import dk.app.core.StateKind;

public abstract class JobExcecutorBase implements IntfWFDExecutor {

	protected IntfEnv env=null;
	protected RunMode runMode= RunMode.SYNC;
	protected StateKind myState=StateKind.INITED; //hacking
	//private JobFile jobFile=null;
	public IntfEnv getEnv() {
		return env;
	}

	public Object getResult() {
		// TODO Auto-generated method stub
		return null;
	}

	public RunMode getRunMode() {
		return runMode;
	}

	public StateKind getState() {
		return myState;
	}

	public boolean interpret(IntfWFD wfd) {
		return false;
	}

	public boolean isCompatible(IntfEnv e) {
		return true;
	}

	
	public void run(IntfEnv env, IntfWFD wfd) {
		setEnv(env);
		interpret(wfd);
		run();
	}

	public void setEnv(IntfEnv env) {
		// TODO Auto-generated method stub
		this.env=env;
	}

	public RunMode setRunMode(RunMode rm) {
		return runMode;
	}


}
