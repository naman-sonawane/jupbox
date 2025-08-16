import { useState, useEffect } from 'react';

export const useFormHydration = () => {
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  const getFormProps = (props: any) => {
    if (!isHydrated) {
      return {
        ...props,
        suppressHydrationWarning: true,
        'data-hydration-safe': 'true'
      };
    }
    return props;
  };

  return { isHydrated, getFormProps };
};
