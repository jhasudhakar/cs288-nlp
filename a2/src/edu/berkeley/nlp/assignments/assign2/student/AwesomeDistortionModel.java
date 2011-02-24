package edu.berkeley.nlp.assignments.assign2.student;

import edu.berkeley.nlp.mt.decoder.DistortionModel;

public class AwesomeDistortionModel implements DistortionModel {
  
  /**
   * Decoding took 416.587s
BLEU score on test data was BLEU(24.212)
Total model score on test data was -3371.0790130316364
   */

  int distortionLimit;
  double weight;

  public AwesomeDistortionModel(int distortionLimit, double weight) {
    this.distortionLimit = distortionLimit;
    this.weight = weight;
  }

  /*
   * (non-Javadoc)
   * 
   * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#
   * getDistortionLimit()
   */
  public int getDistortionLimit() {
    return this.distortionLimit;
  }

  /*
   * (non-Javadoc)
   * 
   * @see edu.berkeley.nlp.assignments.assign2.solutions.DistortionModel#
   * getDistortionScore(int, int)
   */
  public double getDistortionScore(int endPrev, int beginCurr) {
    int dist = Math.abs(beginCurr - endPrev);
    if (dist > distortionLimit) {
      return Double.NEGATIVE_INFINITY;
    }
    return java.lang.Math.pow(weight, dist) - 1;
  }
}
