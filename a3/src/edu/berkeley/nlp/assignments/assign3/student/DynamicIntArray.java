package edu.berkeley.nlp.assignments.assign3.student;

/**
 * An integer (primitive data type) array that dynamically grows. We don't use
 * the Java Vector class because the auto-boxing of primitives cost too much 
 * overhead.
 * 
 * @author rxin
 */
public class DynamicIntArray {
   
   private int[] array = null;
   
   public DynamicIntArray(int init_size) {
      array = new int[init_size];
   }
   
   public int get(int index) {
      return array[index];
   }
   
   public void set(int index, int value) {
      if (index >= array.length) {
         grow(array.length * 2);
      }
      array[index] = value;
   }
   
   public void fastSet(int index, int value) {
      array[index] = value;
   }
   
   public void inc(int index, int value) {
      if (index >= array.length) {
         grow(array.length * 2);
      }
      array[index] += value;
   }
   
   public int length() {
      return array.length;
   }
   
   public void grow(int new_size) {
      int[] new_array = new int[new_size];
      System.arraycopy(array, 0, new_array, 0, array.length);
      array = new_array;
   }

}
