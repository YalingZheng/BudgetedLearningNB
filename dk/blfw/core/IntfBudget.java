package dk.blfw.core;

public interface IntfBudget {
	/**
	 * the initial budget in double
	 * @return
	 */
	double getBudget();
	
	/**
	 * current money left in double
	 * @return
	 */
	double left();
	
	/**
	 * spend money on obj, return true if successful, otherwise return false;
	 * what is left is not changed.
	 * @param obj
	 * @return
	 */
	boolean spendOn(Object obj);
	
	/**
	 * spend d unit of budget, return left if successful, otherwise return null;
	 * what is left is not changed.
	 * @param d
	 * @return
	 */
	boolean spend(double d);
}
