package dk.blfw.alg;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class RowSelector{
    Random r; 
    
    public RowSelector(){
    	r = new Random();
        
    }
    
    public void setSeed(long seed){
    	r.setSeed(seed);
    }
    
    
    public double MyXLogX(double x)
    {
    	//Utils.xlogx(x);
    	double precision = 1000000.0; //6 decimal places
    	double d = Math.floor(x * precision) / precision;
    	double MyD = 0.0;
    	if(d < 0.0 || d > 1.0)
    		System.err.println("Error: MyXLogX(x): x = "+x+" is out of range.");
    	else if(Double.isInfinite(d))
    		System.err.println("Error: MyXLogX(x): x = "+x+" is infinite.");
    	else if(Double.isNaN(d))
    		System.err.println("Error: MyXLogX(x): x = "+x+" is NaN.");
    	else if(d == 0.0 || d == 1.0)
    		return 0.0;
    	else
    		MyD = d*Utils.log2(d);
		return MyD;
    }
    
    public int SelectRow_First(Instances pool, int desiredAttr, int desiredLabel){
        //buy the desiredAttr-th attribute of an (the first) instance with label argmin_j; 
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) )
                return i;
        }
        return -1;
    }
    
    public int SelectRow_First(Instances pool, int desiredAttr){
        //buy the desiredAttr-th attribute of an (the first) instance regardless of label 
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            if(inst.isMissing(desiredAttr))
                return i;
        }
        return -1;
    }

    public int SelectRow_Random(Instances pool, int desiredAttr, int desiredLabel){
        //randomly select among instances with 
        //  -unbought desiredAttr and
        //  -desiredLabel
        int numberValidInstances = 0;
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) )
                numberValidInstances++;
        }
        if(numberValidInstances == 0)
        	return -1;
        int randomInstance = r.nextInt(numberValidInstances);
        int index = 0;
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) )
            {
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }

    public int SelectRow_Random(Instances pool, int desiredAttr){
        //randomly select among instances with 
        //  -unbought desiredAttr and
        //  -desiredLabel
        int numberValidInstances = 0;
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            
            if(inst.isMissing(desiredAttr) )
                numberValidInstances++;
        }
        if(numberValidInstances == 0)
        	return -1;
        int randomInstance = r.nextInt(numberValidInstances);
        int index = 0;
        for(int i=0; i<pool.numInstances(); i++)
        {           
            Instance inst = pool.instance(i);
            if(inst.isMissing(desiredAttr) )
            {
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }   

    public int SelectRow_KLDivergence(Instances pool, Classifier myEstimator, int desiredAttr, int desiredLabel){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure KL-divergence (relative entropy between two prob distributions):
    	//  KL(P||Q) = sum_i  p_i log (p_i/q_i)
    	// withr respect to Q = Uniform, we have
    	//  KL(P||U) = sum_i p_i log(p_i)
    	// choose (row) that is minimum (i.e. closest to uniform)
    	
    	int numInstances = pool.numInstances();
    	double[] KLDivs = new double[numInstances];
    	boolean[] isValidInstance = new boolean[numInstances];
    	double[] probs = null;
    	Instance inst;

    	for(int i=0; i<numInstances; i++)
    	{
    		inst = pool.instance(i);
    		if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) )
    		{
	    		try {
					probs = myEstimator.distributionForInstance(pool.instance(i));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	for(int j=0; j<probs.length; j++)
	        		KLDivs[i] += MyXLogX(probs[j]);
	        	isValidInstance[i] = true;
    		}
    		else
    		{
    			KLDivs[i] = Double.MAX_VALUE;
    			isValidInstance[i] = false;
    		}
    	}
    	
    	double leastDivergence = KLDivs[Utils.minIndex(KLDivs)];
    	int numLeastDivs = 0;
        for(int i=0; i<numInstances; i++)
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        		numLeastDivs++;
        int randomInstance = r.nextInt(numLeastDivs);
        int index = 0;
        for(int i=0; i<numInstances; i++)
        {           
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        	{
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }

    public int SelectRow_KLDivergenceMisclassified(Instances pool, Classifier myEstimator, int desiredAttr, int desiredLabel){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure KL-divergence (relative entropy between two prob distributions):
    	//  KL(P||Q) = sum_i  p_i log (p_i/q_i)
    	// withr respect to Q = Uniform, we have
    	//  KL(P||U) = sum_i p_i log(p_i)
    	// choose (row) that is minimum (i.e. closest to uniform)
    	
    	int numInstances = pool.numInstances();
    	double[] KLDivs = new double[numInstances];
    	boolean[] isValidInstance = new boolean[numInstances];
    	boolean misclassified = false;
    	double[] probs = null;
    	Instance inst;

    	for(int i=0; i<numInstances; i++)
    	{
    		inst = pool.instance(i);
    		try {
				if(inst.classValue() != myEstimator.classifyInstance(inst))
					misclassified = true;
				else
					misclassified = false;
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
    		if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) && misclassified)
    		{
	    		try {	    			
	    		    probs = myEstimator.distributionForInstance(inst);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	for(int j=0; j<probs.length; j++)
	        		KLDivs[i] += MyXLogX(probs[j]);
	        	isValidInstance[i] = true;
    		}
    		else
    		{
    			KLDivs[i] = Double.MAX_VALUE;
    			isValidInstance[i] = false;
    		}
    	}
    	
    	double leastDivergence = KLDivs[Utils.minIndex(KLDivs)];
    	int numLeastDivs = 0;
        for(int i=0; i<numInstances; i++)
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        		numLeastDivs++;
        int randomInstance = r.nextInt(numLeastDivs);
        int index = 0;
        for(int i=0; i<numInstances; i++)
        {           
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        	{
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }

    public int SelectRow_KLDivergenceMisclassified(Instances pool, Classifier myEstimator, int desiredAttr){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure KL-divergence (relative entropy between two prob distributions):
    	//  KL(P||Q) = sum_i  p_i log (p_i/q_i)
    	// withr respect to Q = Uniform, we have
    	//  KL(P||U) = sum_i p_i log(p_i)
    	// choose (row) that is minimum (i.e. closest to uniform)
    	
    	int numInstances = pool.numInstances();
    	double[] KLDivs = new double[numInstances];
    	boolean[] isValidInstance = new boolean[numInstances];
    	boolean misclassified = false;
    	double[] probs = null;
    	Instance inst;

    	for(int i=0; i<numInstances; i++)
    	{
    		inst = pool.instance(i);
    		try {
				if(inst.classValue() != myEstimator.classifyInstance(inst))
					misclassified = true;
				else
					misclassified = false;
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
    		if(inst.isMissing(desiredAttr) && misclassified)
    		{
	    		try {	    			
	    		    probs = myEstimator.distributionForInstance(inst);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	for(int j=0; j<probs.length; j++)
	        		KLDivs[i] += MyXLogX(probs[j]);
	        	isValidInstance[i] = true;
    		}
    		else
    		{
    			KLDivs[i] = Double.MAX_VALUE;
    			isValidInstance[i] = false;
    		}
    	}
    	
    	double leastDivergence = KLDivs[Utils.minIndex(KLDivs)];
    	int numLeastDivs = 0;
        for(int i=0; i<numInstances; i++)
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        		numLeastDivs++;
        int randomInstance = r.nextInt(numLeastDivs);
        int index = 0;
        for(int i=0; i<numInstances; i++)
        {           
        	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
        	{
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }

    public int SelectRow_KLDivergence(Instances pool, Classifier myEstimator, int desiredAttr){

        	//for each instance with unbought desiredAttr and label = desiredLabel
        	// measure KL-divergence (relative entropy between two prob distributions):
        	//  KL(P||Q) = sum_i  p_i log (p_i/q_i)
        	// withr respect to Q = Uniform, we have
        	//  KL(P||U) = sum_i p_i log(p_i)
        	// choose (row) that is minimum (i.e. closest to uniform)
        	
        	int numInstances = pool.numInstances();
        	double[] KLDivs = new double[numInstances];
        	boolean[] isValidInstance = new boolean[numInstances];
        	double[] probs = null;
        	Instance inst;

        	for(int i=0; i<numInstances; i++)
        	{
        		inst = pool.instance(i);
        		if(inst.isMissing(desiredAttr))
        		{
    	    		try {
    					probs = myEstimator.distributionForInstance(pool.instance(i));
    				} catch (Exception e) {
    					// TODO Auto-generated catch block
    					e.printStackTrace();
    				}
    	        	for(int j=0; j<probs.length; j++)
    	        		KLDivs[i] += MyXLogX(probs[j]);
    	        	isValidInstance[i] = true;
        		}
        		else
        		{
        			KLDivs[i] = Double.MAX_VALUE;
        			isValidInstance[i] = false;
        		}
        	}
        	
        	double leastDivergence = KLDivs[Utils.minIndex(KLDivs)];
        	int numLeastDivs = 0;
            for(int i=0; i<numInstances; i++)
            	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
            		numLeastDivs++;
            int randomInstance = r.nextInt(numLeastDivs);
            int index = 0;
            for(int i=0; i<numInstances; i++)
            {           
            	if(isValidInstance[i] && KLDivs[i] == leastDivergence)
            	{
                    if(index == randomInstance)
                        return i;
                    else 
                        index++;
                }
            }
            return -1;
        }

        public int SelectRow_L2Norm(Instances pool, Classifier myEstimator, int desiredAttr, int desiredLabel){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure distance from uniform
    	// choose (row) that is closest to uniform as your instance to buy from  	

    	double leastDistance = Double.MAX_VALUE;
    	int leastIndex = -1;
    	Instance inst;
    	int n = pool.numClasses();
    	double[] uniform;
    	double[] probs;
    	uniform = new double[n];
    	for(int i=0; i<n; i++)
    		uniform[i] = 1.0 / (double) n;
    	
        for(int i=0; i<pool.numInstances(); i++)
        {           
            inst = pool.instance(i);
            //System.out.println("currentlabel="+(int)inst.classValue()+" isMissing="+inst.isMissing(desiredAttr));
            if( (int) inst.classValue() == desiredLabel && inst.isMissing(desiredAttr) )
            {
            	//valid instance
            	//measure the distance from uniform:
            	//sqrt{ sum_i (a_i - b_i)^2 }
            	probs = new double[n];
            	try {
					probs = myEstimator.distributionForInstance(inst);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	double distance = 0.0;
            	for(int j=0; j<n; j++)
            		distance += (probs[j] - uniform[j]) * (probs[j] - uniform[j]);
            	distance = Math.sqrt(distance);      
            	//System.out.println("current distance="+distance);
            	if(distance < leastDistance)
            	{
            		leastDistance = distance;
            		leastIndex = i;
            	}
            	
            }
        }
        return leastIndex;
    }
    
    public int SelectRow_L2Norm(Instances pool, Classifier myEstimator, int desiredAttr){
        
    	//for each instance with unbought desiredAttr and any label 
    	// measure distance from uniform
    	// choose (row) that is closest to uniform as your instance to buy from
    	double leastDistance = Double.MAX_VALUE;
    	int leastIndex = -1;
    	Instance inst;
    	int n = pool.numClasses();
    	double[] uniform;
    	double[] probs;
    	uniform = new double[n];
    	for(int i=0; i<n; i++)
    		uniform[i] = 1.0 / (double) n;
    	
        for(int i=0; i<pool.numInstances(); i++)
        {           
            inst = pool.instance(i);
            if(inst.isMissing(desiredAttr) )
            {
            	//valid instance
            	//measure the distance from uniform:
            	//sqrt{ sum_i (a_i - b_i)^2 }
            	probs = new double[n];
            	try {
					probs = myEstimator.distributionForInstance(inst);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	double distance = 0.0;
            	for(int j=0; j<n; j++)
            		distance += (probs[j] - uniform[j]) * (probs[j] - uniform[j]);
            	distance = Math.sqrt(distance);            	
            	if(distance < leastDistance)
            	{
            		leastDistance = distance;
            		leastIndex = i;
            	}
            	
            }
        }
        return leastIndex;
    }
    
    int SelectRow_ErrorMargin(Instances pool, Classifier myEstimator, int desiredAttr, int desiredLabel){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure Prob(i,L(i)) the class probability of the true label, choose the one minimizing it.
    	// i.e. the most erroneous instance
    	
    	int numInstances = pool.numInstances();
    	double[] classProb = new double[numInstances];
    	boolean[] isValidInstance = new boolean[numInstances];
    	double[] probs = null;
    	Instance inst;

    	for(int i=0; i<numInstances; i++)
    	{
    		inst = pool.instance(i);
    		if(inst.isMissing(desiredAttr)&& inst.classValue()==desiredLabel )
    		{
	    		try {	    			
	    		    probs = myEstimator.distributionForInstance(inst);
	    		    classProb[i]=probs[desiredLabel];
	    		    isValidInstance[i] = true;
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	
    		}
    		else
    		{
    			classProb[i] = Double.POSITIVE_INFINITY;
    			isValidInstance[i] = false;
    		}
    	}
    	
    	double leastCorrect = classProb[Utils.minIndex(classProb)];
    	int numLeastCorrect = 0;
        for(int i=0; i<numInstances; i++)
        	if(isValidInstance[i] && classProb[i] == leastCorrect)
        		numLeastCorrect++;
        int randomInstance = r.nextInt(numLeastCorrect);
        int index = 0;
        for(int i=0; i<numInstances; i++)
        {           
        	if(isValidInstance[i] && classProb[i] == leastCorrect)
        	{
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }
    
    int SelectRow_ErrorMargin(Instances pool, Classifier myEstimator, int desiredAttr){

    	//for each instance with unbought desiredAttr and label = desiredLabel
    	// measure Prob(i,L(i)) the class probability of the true label, choose the one minimizing it.
    	// i.e. the most erroneous instance
    	
    	int numInstances = pool.numInstances();
    	double[] classProb = new double[numInstances];
    	boolean[] isValidInstance = new boolean[numInstances];
    	double[] probs = null;
    	Instance inst;

    	for(int i=0; i<numInstances; i++)
    	{
    		inst = pool.instance(i);
    		if(inst.isMissing(desiredAttr) )
    		{  
	    		try {	    			
	    		    probs = myEstimator.distributionForInstance(inst);
	    		    classProb[i]=probs[(int)inst.classValue()];
	    		    isValidInstance[i]=true;
	    		    
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	
    		}
    		else
    		{
    			classProb[i] = Double.POSITIVE_INFINITY;
    			isValidInstance[i] = false;
    		}
    	}
    	
    	
    	double leastCorrect = classProb[Utils.minIndex(classProb)];
    	int numLeastCorrect = 0;
        for(int i=0; i<numInstances; i++){
        	if(isValidInstance[i] && classProb[i] == leastCorrect)
        		numLeastCorrect++;
        }
        
     
        int randomInstance = r.nextInt(numLeastCorrect);
        int index = 0;
        
        for(int i=0; i<numInstances; i++)
        {           
        	if(isValidInstance[i] && classProb[i] == leastCorrect)
        	{
                if(index == randomInstance)
                    return i;
                else 
                    index++;
            }
        }
        return -1;
    }
    
    

}
