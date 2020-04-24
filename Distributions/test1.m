%function test1(xx,AA)


%TEST1    Create plot of datasets and fits
%   TEST1(XX,AA)
%   Creates a plot, similar to the plot in the main curve fitting
%   window, using the data that you provide as input.  You can
%   apply this function to the same data you used with cftool
%   or with different data.  You may want to edit the function to
%   customize the code and this help message.
%
%   Number of datasets:  1
%   Number of fits:  1

 
% Data from dataset "AA vs. xx":
%    X = xx:
%    Y = AA:
%    Unweighted
%
% This function was automatically generated on 09-Aug-2012 11:52:14

% Set up figure to receive datasets and fits
f_ = clf;
figure(f_);
set(f_,'Units','Pixels','Position',[494 181 680 484]);
legh_ = []; legt_ = {};   % handles and text for legend
xlim_ = [Inf -Inf];       % limits of x axis
ax_ = axes;
set(ax_,'Units','normalized','OuterPosition',[0 0 1 1]);
set(ax_,'Box','on');
axes(ax_); hold on;

 
% --- Plot data originally in dataset "AA vs. xx"
xx = xx(:);
AA = AA(:);
h_ = line(xx,AA,'Parent',ax_,'Color',[0.333333 0 0.666667],...
     'LineStyle','none', 'LineWidth',1,...
     'Marker','.', 'MarkerSize',12);
xlim_(1) = min(xlim_(1),min(xx));
xlim_(2) = max(xlim_(2),max(xx));
legh_(end+1) = h_;
legt_{end+1} = 'AA vs. xx';

% Nudge axis limits beyond data limits
if all(isfinite(xlim_))
   xlim_ = xlim_ + [-1 1] * 0.01 * diff(xlim_);
   set(ax_,'XLim',xlim_)
end


% --- Create fit "fit 1"
ok_ = ~(isnan(xx) | isnan(AA));
ft_ = fittype('poly1');

% Fit this model using new data
cf_ = fit(xx(ok_),AA(ok_),ft_);

% Or use coefficients from the original fit:
if 0
   cv_ = {0.1432848182517, -0.7867898922528};
   cf_ = cfit(ft_,cv_{:});
end

% Plot this fit
h_ = plot(cf_,'fit',0.95);
legend off;  % turn off legend from plot method call
set(h_(1),'Color',[1 0 0],...
     'LineStyle','-', 'LineWidth',2,...
     'Marker','none', 'MarkerSize',6);
legh_(end+1) = h_(1);
legt_{end+1} = 'fit 1';

% Done plotting data and fits.  Now finish up loose ends.
hold off;
h_ = legend(ax_,legh_,legt_,'Location','NorthEast');  
set(h_,'Interpreter','none');
xlabel(ax_,'');               % remove x label
ylabel(ax_,'');               % remove y label
