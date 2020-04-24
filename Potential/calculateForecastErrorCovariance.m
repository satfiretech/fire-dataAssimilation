function K = calculateForecastErrorCovariance(EnsPX_)

K = [];
Ne = size(EnsPX_,3);
for pixel = 1:size(EnsPX_,2)
    SEnsPX_ = zeros(size(EnsPX_,1),size(EnsPX_,1));
    for member = 1:Ne
        SEnsPX_ = SEnsPX_ + EnsPX_(:,pixel,member)*EnsPX_(:,pixel,member)';
    end
    K = [K SEnsPX_/(Ne - 1)];
end


