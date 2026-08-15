import type { ButtonHTMLAttributes } from 'react';

// Reusable pill button. By default it renders with the `.primary` class so it
// is visually identical to the existing app buttons (blue, rounded, with the
// CSS-defined hover/disabled states). Pass `color` to override the background
// for variants the upcoming talk mode needs — when omitted, no inline style is
// applied so the CSS `:hover`/`:disabled` rules keep working unchanged.
type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  color?: string;
};

export function Button({ color, className, style, ...rest }: ButtonProps) {
  return (
    <button
      className={className ?? 'primary'}
      style={color ? { background: color, ...style } : style}
      {...rest}
    />
  );
}
