package dk.blfw.impl;

import weka.core.Instance;


/**
 * 
 * @author kdeng
 *
 */
public interface IntfAttrUpdatable {
	/**
	 * update the count for instance inst with attribute attIndex;
	 * currently only handle one case: inclusion of attribute value that wasn't there before.
	 * to remove the effect of some attr (exclusion): the estimator has to be extended. 
	 * this method should not be called more than once with non-missing attribute.  
	 * @param inst
	 * @param attIndex
	 */
	void updateClassifierAttribute(Instance inst, int attIndex);
	
	
}
