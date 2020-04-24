function Error = derror(ThresholdCoefficient,FAR,p)

x = FAR(:);
ThresholdCoefficient = ThresholdCoefficient(:);

EstimatedThresholdCoefficient = p(1)*log(x) + p(2)*x + p(3);

Error = mean((ThresholdCoefficient - EstimatedThresholdCoefficient).^2);
