# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Logistic Regression classifier from the Scikit-learn library. Used sklearn.linear_model.LogisticRegression. model path: starter/starter/ml/model.py

## Intended Use
The model is intended to predict whether an individual's income is above or below $50K based on census data for educational purposes.

## Training Data
The project uses the "Census Income" dataset. And it's from the UCI Machine Learning Repository. Performed cleaning steps like removing spaces.

## Evaluation Data
path: starter/starter/train_model.py. The model was evaluated on a 20% hold-out test set that was not used during training.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7285, Recall: 0.2699, F1 Score: 0.3939
	
## Ethical Considerations
An analysis of the model's performance on slices of the data reveals significant performance disparities across different groups, indicating a potential for harmful bias. When evaluating the model against the 'education' feature, there is a clear trend where the model's predictive accuracy is much higher for individuals with higher levels of formal education. For instance, the model achieves a relatively high F1-score for individuals with a 'Prof-school' (0.50) or 'Doctorate' (0.50) level of education. In contrast, its performance is substantially worse for those with lower education levels, such as '7th-8th' (F1-score of 0.20) and '12th' (0.22). Most notably, the model fails completely for the '1st-4th' category, achieving an F1-score of 0.0, meaning it was unable to correctly identify any high-income individuals in this group. If used in a real-world application, this model would be less reliable and potentially unfair to individuals with less formal education, systematically under-predicting their economic standing. Given that this bias exists on the 'education' feature, it is highly probable that similar performance disparities exist for other sensitive features like race and sex. Therefore, deploying this model for any real-world decision-making could reinforce and amplify existing societal inequities. Due to these identified biases, the model is not recommended for use outside of this educational demonstration.

## Caveats and Recommendations
The model is trained on historical data and may not perform as well on future data. This model is for demonstration only and should not be used for real-world decisions like loan approvals without a much more thorough fairness audit.