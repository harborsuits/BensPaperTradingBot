import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

import { Play } from 'lucide-react';
import { Link } from 'react-router-dom';

const EnhancedDecisionCard = ({ decision, className = '' }) => {
  return (
    <Card className={`flex flex-col ${className}`}>
      <CardContent className="p-4 space-y-3 text-sm">
        <div>
          <p className="text-muted-foreground text-xs">What we did</p>
          <p className="font-medium">{decision.what}</p>
        </div>
        <div>
          <p className="text-muted-foreground text-xs">Why</p>
          <p>{decision.why}</p>
        </div>
        <div>
          <p className="text-muted-foreground text-xs">Risk plan</p>
          <p>{decision.riskPlan}</p>
        </div>
        <div>
          <p className="text-muted-foreground text-xs">Result so far</p>
          <p>{decision.resultSoFar}</p>
        </div>
      </CardContent>
      <CardFooter className="p-4 mt-auto bg-muted/50 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-muted-foreground">Trace & Replay</span>
          <Badge variant="secondary">{decision.tradeIdPill}</Badge>
        </div>
        <Button size="sm" asChild>
          <Link to={`/trace/${decision.id}/replay`}>
            <Play size={14} className="mr-1" /> Replay
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
};

export default EnhancedDecisionCard;
