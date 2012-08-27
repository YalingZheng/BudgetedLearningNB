package dk.blfw.impl;

import dk.blfw.core.IntfBudget;

public class PhonySimpleBudget implements IntfBudget {
	private double budget;
	private double left;

	public PhonySimpleBudget(double b){
		setBudget(b);
	}
	
	public double getBudget() {
		return budget;
	}
	
	protected void setBudget(double b) {
		budget= b;
		left=budget;
		
	}

	public double left() {
		return left;
	}

	/**
	 * there are holes left on purpose:
	 * can increase budget by spend negative
	 * or  just override this method in subclasses;
	 * can spend half unit, by design.
	 */
	public boolean spend(double unit) {
		if (left-unit>=0)
		{
			left-=unit;
			return true;
			
		}
		
		return false;
	}

	public boolean spendOn(Object obj) {
		
		if (!spend(1.0) )
		return false;
		return true;
	}

}
