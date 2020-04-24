function KeepScaledResidual = setFireSampleToMissing(KeepScaledResidual,KeepFireDetect)

KeepScaledResidual(KeepFireDetect==1) = nan;
