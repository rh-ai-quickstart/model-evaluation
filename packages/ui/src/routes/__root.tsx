// ai-quickstart-template - Root Route
import { createRootRoute, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import { Toaster } from 'sonner';
import { Header } from '../components/header/header';
import { Footer } from '../components/footer/footer';

export const Route = createRootRoute({
  component: () => (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
      <Footer />
      <Toaster richColors position="bottom-right" />
      {import.meta.env.DEV && <TanStackRouterDevtools />}
    </div>
  ),
});