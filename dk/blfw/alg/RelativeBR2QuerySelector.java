/* BR2QuerySelector
 * Input: 
 * Output: an attribute and an instance 
 * 
 * Procedure:
 * if latest change (of instance classification), i.e., the summation
 * of the change of classification of each instance, is significant 
 * (and not all the instances of an attribute has been chosen), we
 * continue to choose this attribute, otherwise, we switch to the next
 * attribute. Once an attribute is finalized, we arbitrarily choose an
 * instance. 
 * 
 * Note in this program, I set alpha to check whether a change is significant 
 * or not. 
 * */

package dk.blfw.alg;
import java.lang.Math;

//import java.util.Arrays;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
//import weka.estimators.Estimator;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;
import dk.blfw.impl.NaiveBayesAttrUpdatable;

import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.FastVector;

public class RelativeBR2QuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2L;
	private static final int seed=1234;
	private Random random= new Random(seed);
	
	private int[] permutation;
	private boolean inited=false;
	// alpha is the value to judge whether it is a signficant change
	private double alpha=0.3;
	private double lastdifference = 0.0;
	// lastai saves the index of last chosen attribute
	private int lastaiindex=-1;
	// lastclassesprob saves the classification for each instance;
	private double[][] lastclassesprob;
	private int[] numinstanceseachattribute;
	
	private int numinstances;
	private int numclasses;
	private int numattributes;
//	private int counter=0;
	@Override
	public String[] getOptions() {
	    String [] options;
	    options = new String[1];
	    options[0]="-alpha:"+alpha;
	    return options;
	}

	@Override
	@SuppressWarnings("unchecked")
	public Enumeration listOptions() {

	    Vector newVector = new Vector(1);

	    newVector.addElement(new Option(
		      "\talpha constant\n"
		      + "\t that defines the relative change",
		      "alpha", 1, "-alpha <value>"));
	    return newVector.elements();
	}


	@Override
	public void setOptions(String[] options) throws Exception {
		if (Utils.getOptionPos("alpha", options)>=0){
			 
			String tmp= (Utils.getOption("alpha", options));
			double g= Double.parseDouble(tmp);
			setAlpha(g);
	
		}
		
	}

	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}



	public double getAlpha() {
		return alpha;
	}	


	/**
	 * 
	 * @param g parameter \alpha for AbsoluteBR2Query
	 */
	public RelativeBR2QuerySelector(double g){
		super();
		setAlpha(g);
	}
	
	public RelativeBR2QuerySelector() {
		super();
	}
	
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		NaiveBayesAttrUpdatable c= (NaiveBayesAttrUpdatable) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		//System.out.println("run");
		//System.out.println(counter++);
		numinstances=pool.numInstances();
		numclasses = pool.numClasses();
		numattributes = pool.numAttributes()-1;
		
		if (!inited){
			inited=true;
			// allocate space to lastclassesprob;
			
			//generate a permutation
			permutation = new int[numattributes];
			permutation = get_curent_permutation(numattributes);
			
			// initialize lastclassesprob
			lastclassesprob = new double[numinstances][numclasses];
			for (int i=0;i<numinstances;i++){
				for (int j=0;j<numclasses;j++){
					lastclassesprob[i][j]=0;
				}
			}
			numinstanceseachattribute = new int[numattributes];
			for (int j=0; j<numattributes;j++){
				numinstanceseachattribute[j] = numinstances;
			}
			
		
		}	
		
		// the following, we need to 
		// choose a new attribute and a new instance
		
		// if lastai = -1, then it is the first time we choose an attribute
		if (lastaiindex>=0){
			//receive reward for last action
			//update

			try {

				calclNew(pool,c);
				// get new classesprob[i][j]=0;
	
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		// choose an instance, if 
		// all instances of this attribute are chosen
		// or
		// there is no significant change on the classification
		// then we choose the next attribute
		
		/**
		 * quit util find a missing attribute
		 */
		int aiindex=lastaiindex;
		if (aiindex<0){aiindex=0;}
		if ((numinstanceseachattribute[permutation[aiindex]]>0)&&(significantchange(alpha,pool,c))){
			// continue to consider the same attribute
			System.out.println("Enter Place 1");
		   while(true){
			int ii= drawInst(pool, c);
	        //System.out.println("loop " + ai + " " + ii);
	        if  (pool.instance(ii).isMissing(permutation[aiindex])) {
		        lastaiindex=aiindex;
		        //System.out.println("(i,a)=("+ii+","+ai+")");
		        numinstanceseachattribute[permutation[aiindex]]--;
		        System.out.println("instance "+ii+" attribute "+permutation[aiindex]);
		        return new PhonyAttrQuery(ii,permutation[aiindex]);}
			}
		   }					
		else {
			System.out.println("Enter Place else..............");
	  // else, force to consider next attribute
			while (true){
				aiindex = drawAttr(numattributes);
		    	while(true){
					int ii= drawInst(pool, c);
			        if  (pool.instance(ii).isMissing(permutation[aiindex])) {
				        lastaiindex=aiindex;
				        numinstanceseachattribute[permutation[aiindex]]--;
				        System.out.println("instance "+ii+" attribute "+permutation[aiindex]);
				        return new PhonyAttrQuery(ii,permutation[aiindex]);}
					}	
				}
		}
   }
	
	private int[] get_curent_permutation(int numattributes){
	 	int[] cur_permutation = new int[numattributes];
	 	for (int i=0;i<numattributes;i++)
	 	{cur_permutation[i] = i;}
	 	/*int index1=0;
	 	int index2=0;
	 	permutationrandom.setSeed(dk.blfw.Global.permutationseed);
	 	for (int i=0;i<permutationrandom.nextInt(10000)+1000;i++){
	 		index1 = permutationrandom.nextInt(numattributes-1);
	 		index2 = permutationrandom.nextInt(numattributes-1);
	 		int temp = cur_permutation[index1];
	 		cur_permutation[index1]=cur_permutation[index2];
	 		cur_permutation[index2]=temp;
	 	}*/
	 	//System.out.println();
	 	//for (int i=0;i<numattributes;i++){
	 	//	System.out.print(cur_permutation[i]+" ");
	 	//}
	 	//System.out.println();
		return cur_permutation;
		
	}
	double calclNew(Instances pool, NaiveBayesAttrUpdatable c) throws Exception {
		
		EvaluationUtils eu= new EvaluationUtils();
		String[]	classnames= new String[pool.classAttribute().numValues()];
		for(int i=0; i<classnames.length; i++){
			classnames[i]= "class"+ Integer.toString(i);
		}
		
		FastVector preds=eu.getTestPredictions(c, pool);
		
		ConfusionMatrix cm= new ConfusionMatrix(classnames);
		
		cm.addPredictions(preds);
		
		double x_i= cm.errorRate();
		return x_i;
	
	}
	
	// judge whether it is a significant change
	private boolean significantchange(double alpha,Instances pool,NaiveBayesAttrUpdatable c){
		int numinstances = pool.numInstances();
		int numclasses = pool.numClasses();
		double[][] currentclassesprob = new double[numinstances][numclasses];
		double[] cur_instance_prob = new double[numclasses];
		double sum_difference = 0;
		for (int i=0;i<numinstances;i++){
			try{
		       cur_instance_prob = c.distributionForInstance(pool.instance(i));
		       for (int j=0;j<numclasses;j++){
		    	   currentclassesprob[i][j]=cur_instance_prob[j];
		    	   //System.out.println("["+i+","+j+"]="+currentclassesprob[i][j]);
		    	   sum_difference += Math.abs(currentclassesprob[i][j]-lastclassesprob[i][j]);
		    	   lastclassesprob[i][j]=currentclassesprob[i][j];
    	   		       }
			}
			catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();}
			
		}
		System.out.println("last difference = " + lastdifference);
		System.out.println("current difference = " + sum_difference);
		System.out.println("ratio = " + sum_difference/lastdifference);
		double oldlastdifference = lastdifference;
		lastdifference = sum_difference;
		if (oldlastdifference == 0)
			return true;
		else {
		  if ((sum_difference/oldlastdifference) > alpha)
			  return true;
		  else
			  return false;
		}		
		
	}
	
	private int drawAttr(int numAttributes){
	  while (true){
		 lastaiindex = (lastaiindex + 1) % numAttributes;
		 if (numinstanceseachattribute[permutation[lastaiindex]]>0)
		    {
		    	return lastaiindex;
		    }
	     }
	}
	
	private int drawInst(Instances p, NaiveBayesAttrUpdatable c){
		double rand = dk.blfw.Global.random.nextDouble();
		int myLabel = 0;
		int i = 0;
		double soFar = 0;
		boolean found = false;
		while(!found)
		{
			soFar += c.getClassDistribution().getProbability(i);
			if(rand <= soFar)
			{
				myLabel = i;
				found = true;
			}
			i++;
		}
			
		int j = -1;
		while(true)
		{
			j = dk.blfw.Global.random.nextInt(p.numInstances());
			if  (p.instance(j).classValue() == myLabel) {
				return j;
			}	
		}
	}
	/*private int drawInst(Instances p, NaiveBayesAttrUpdatable c){
		return random.nextInt(p.numInstances());
	}*/
	
	@SuppressWarnings("unchecked")
	private boolean hasMissing(Instances pool){
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.hasMissingValue()) {hasmissing=true; break;}
		}
		return hasmissing;
	}
}
