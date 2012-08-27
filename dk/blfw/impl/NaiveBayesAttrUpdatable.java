package dk.blfw.impl;


import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.estimators.*;

public class NaiveBayesAttrUpdatable extends NaiveBayes implements IntfAttrUpdatable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public void updateClassifierAttribute(Instance instance, int attIndex) {
		
//		if (!instance.classIsMissing()) {
//		      Enumeration enumAtts = m_Instances.enumerateAttributes();
//		      int attIndex = 0;
//		      while (enumAtts.hasMoreElements()) {
//			Attribute attribute = (Attribute) enumAtts.nextElement();
//			if (!instance.isMissing(attribute)) {
//			  m_Distributions[attIndex][(int)instance.classValue()].
//			    addValue(instance.value(attribute), instance.weight());
//			}
//			attIndex++;
//		      }
//		      m_ClassDistribution.addValue(instance.classValue(),
//						   instance.weight());
//		    }
	
		
		if (!instance.isMissing(attIndex)){
			  m_Distributions[attIndex][(int)instance.classValue()].
		    addValue(instance.value(attIndex), instance.weight());
		}
	
	}
	
	public Estimator getClassDistribution() //Discrete Estimator
	{
		return m_ClassDistribution;
	}

	public Estimator[][] getMDistributions()
	{
		return m_Distributions;
	}

	

	

	
	
}
