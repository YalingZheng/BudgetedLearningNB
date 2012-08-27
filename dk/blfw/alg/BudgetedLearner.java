package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Vector;

import dk.blfw.core.IntfBudget;
import dk.blfw.core.IntfOracle;
import dk.blfw.core.IntfQuery;
import dk.blfw.core.IntfQuerySelector;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.IntfAttrUpdatable;
import dk.blfw.impl.IntfTrainingProgressListener;
import dk.blfw.impl.IntfTrainingProgressProducer;
import dk.blfw.impl.PhonyAttrAnswer;
import dk.blfw.impl.PhonyAttrOracle;
import dk.blfw.impl.PhonyAttrQuery;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * a learner must strictly follow procedures as following:
 * initialize by calling initialize() or constructor;
 * training by calling buildClassifier();
 * testing by calling classifierInstance();
 * 
 * restarting everything by clear();
 * or  reset by cycling the above procedure again.
 * 
 * 
 * @author kdeng
 *
 */


public class BudgetedLearner extends Classifier implements IntfTrainingProgressProducer {
	
	private Vector<IntfTrainingProgressListener> listeners= new Vector<IntfTrainingProgressListener>();
	
	public void register(IntfTrainingProgressListener listener) {
		listeners.add(listener);
	}

	public void unregister(IntfTrainingProgressListener listener) {
		listeners.remove(listener);
	}

	

	enum StateKind{
		UNKNOWN, INITIALIZED, TRAINING, FAILED, READY;
	}
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;



	
	
	private StateKind state=StateKind.UNKNOWN;
	
	private Classifier c;
	private IntfBudget budget;
	private IntfOracle oracle;
	private IntfQuerySelector querySel;
	
	
	public BudgetedLearner(){};
	
	public BudgetedLearner(Classifier base, IntfBudget b, IntfOracle o, IntfQuerySelector qs){
		initialize(base,b, o, qs);
	}
	
	
	public void initialize(Classifier base, IntfBudget b, IntfOracle o, IntfQuerySelector qs){
		if (!(base instanceof IntfAttrUpdatable)){
			System.err.println("warning: base classifier is not attribute updatable.");
		}
		try {
			c=Classifier.makeCopy(base);
		} catch (Exception e) {
			e.printStackTrace();
		}
		budget=b;
		oracle=o;
		querySel=qs;
		state=StateKind.INITIALIZED;
	}

	

	public StateKind getState() {
		return state;
	}
	
	public void clear(){
		state=StateKind.UNKNOWN;
	}
	
	

	@Override
	public void buildClassifier(Instances data) throws Exception {
		if (state!=StateKind.INITIALIZED){
			throw new Exception("Classifier not properly initialized");			
		}
		
		train(data);
		
		
	}
	
	@SuppressWarnings("unchecked")
	public  void train(Instances data) throws Exception {
		state=StateKind.TRAINING;
		//step 0: make sure everything is set up properly.
		if (data.classIndex()<0) throw new Exception("Class Index not set yet!");
//		TODO: data may contain missing attributes as well. Shall we fill in default values?
		
		//make an empty copy of data: remove all attribute values except for the class attribute.
		Instances trainData = new Instances(data);
		for (Enumeration<Instance> e = trainData.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			for(int j=0; j<i.numAttributes(); j++){
				if (j!= i.classIndex()){
					i.setMissing(j);
				}
			}
				
		}
		
		// if oracle not set;  
		if (! (oracle instanceof PhonyAttrOracle))
		{
			oracle= new PhonyAttrOracle(data);
		}
		
		
		//step 1: initialize base classifier with initial data. 
		//TODO: may need add parameter to budgetlearner
		c.buildClassifier(trainData);

		//step 2: train incrementally util budget is exhausted, inform progress listener if necessary
		EnumMap<QueryRequest,Object> context;
		context = new EnumMap(QueryRequest.class);
		
		boolean reportProgress= listeners.size()==0? false:true;
		
		if (reportProgress){
			for(IntfTrainingProgressListener listener: listeners){
				listener.init();
			}
		}
		
		int numIter=0;
		while (budget.spend(1)){
			context.put(QueryRequest.C, c);
			context.put(QueryRequest.P, trainData);
			context.put(QueryRequest.CANDIDATE, trainData);
			IntfQuery query= querySel.propose(context, null);
			if (!(query instanceof PhonyAttrQuery) || (query== IntfQuery.NONQUERY)) {
				throw new Exception("doesn't support this kind of query");
			}
			
			if (query==IntfQuery.NONQUERY){
				break;
			}
			
			PhonyAttrAnswer answer=  (PhonyAttrAnswer) oracle.answerQuery(query);

			if (answer!=null){
				trainData.instance(answer.getIi())
				.setValue(answer.getAi(), answer.getValue());
			}

			if (c instanceof IntfAttrUpdatable) {
				IntfAttrUpdatable cc = (IntfAttrUpdatable) c;
				cc.updateClassifierAttribute(trainData.instance(answer.getIi()), answer.getAi());
			}else{
				c.buildClassifier(trainData);
			}
			
			if (reportProgress){
				for(IntfTrainingProgressListener listener: listeners){
					listener.update(c, trainData, ++numIter);
				}
			}
			
			
		}
		
		if (reportProgress){
			for(IntfTrainingProgressListener listener: listeners){
				listener.done();
			}
		}
			
			
		state=StateKind.READY;	
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		if (state!=StateKind.READY){
			throw new Exception("Classifier not properly trained");
		}
			
		return c.distributionForInstance(instance);
	}




    
	
	
	
	
	
}
