import cv2
import numpy as np


class ImageComparison:
    def __init__(self, original, mosaic):
        self.original = np.array(original)
        self.mosaic = np.array(mosaic)

    def compute_similarity_metrics(self):
      # Resize the images to the same dimensions
      original = cv2.resize(self.original, (self.mosaic.shape[1], self.mosaic.shape[0]))

      # Compute Mean Squared Error (MSE) (lower is better)
      mse_score = np.mean((original - self.mosaic) ** 2)

      # Compute Peak Signal-to-Noise Ratio (PSNR) (higher is better)
      psnr_score = cv2.PSNR(original, self.mosaic)

      # Compute Histogram Comparison (Correlation, Chi-Square, Intersection, Bhattacharyya)
      hist_original = cv2.calcHist([original], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
      hist_mosaic = cv2.calcHist([self.mosaic], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

      hist_original = cv2.normalize(hist_original, hist_original).flatten()
      hist_mosaic = cv2.normalize(hist_mosaic, hist_mosaic).flatten()

      correlation = cv2.compareHist(hist_original, hist_mosaic, cv2.HISTCMP_CORREL)
      chi_square = cv2.compareHist(hist_original, hist_mosaic, cv2.HISTCMP_CHISQR)
      intersection = cv2.compareHist(hist_original, hist_mosaic, cv2.HISTCMP_INTERSECT)
      bhattacharyya = cv2.compareHist(hist_original, hist_mosaic, cv2.HISTCMP_BHATTACHARYYA)

      return {
          "MSE": mse_score,
          "PSNR": psnr_score,
          "Histogram Correlation": correlation,
          "Histogram Chi-Square": chi_square,
          "Histogram Intersection": intersection,
          "Histogram Bhattacharyya": bhattacharyya
      }

    
    def generate_metrics_explanation(self):
        """
        Provides explanations for the computed similarity metrics values.
        Args:
            metrics: Dictionary containing the computed metric values
        Returns:
            String containing human-readable explanations of the metrics
        """
        explanations = []
        
        metrics = self.compute_similarity_metrics()
        # MSE Explanation
        mse = metrics["MSE"]
        explanations.append(f"Mean Squared Error (MSE): {mse:.2f}")
        explanations.append("- Lower values indicate more similarity between images")
        explanations.append("- A value of 0 would mean identical images")
        
        # PSNR Explanation
        psnr = metrics["PSNR"]
        explanations.append(f"\nPeak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
        explanations.append("- Higher values indicate better image quality")
        explanations.append("- Typical values are between 30-50 dB, where higher is better")
        
        # Histogram Correlation Explanation
        correlation = metrics["Histogram Correlation"]
        explanations.append(f"\nHistogram Correlation: {correlation:.4f}")
        explanations.append("- Values range from -1 to 1")
        explanations.append("- 1 indicates perfect correlation, -1 perfect negative correlation")
        
        # Chi-Square Explanation
        chi_square = metrics["Histogram Chi-Square"]
        explanations.append(f"\nHistogram Chi-Square: {chi_square:.4f}")
        explanations.append("- Lower values indicate better matches")
        explanations.append("- 0 would indicate a perfect match")
        
        # Intersection Explanation
        intersection = metrics["Histogram Intersection"]
        explanations.append(f"\nHistogram Intersection: {intersection:.4f}")
        explanations.append("- Higher values indicate better matches")
        
        # Bhattacharyya Explanation
        bhattacharyya = metrics["Histogram Bhattacharyya"]
        explanations.append(f"\nBhattacharyya Distance: {bhattacharyya:.4f}")
        explanations.append("- Lower values indicate better matches")
        explanations.append("- 0 would indicate a perfect match")

        return "\n".join(explanations)