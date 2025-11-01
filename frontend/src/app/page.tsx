'use client';
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col gap-4 items-center justify-center p-24">
      <h1 className="text-2xl font-bold">Welcome to the Home Page</h1>
      <div className="flex flex-row gap-4">
        <Button variant="default" onClick={() => console.log("User Login!!")}>Login</Button>
        <Button variant="outline" onClick={() => console.log("User Register!!")}>Register</Button>
      </div>
    </div>
  );
}
