import { useChatMessage } from "@llamaindex/chat-ui";
import { User2 } from "lucide-react";
import Image from "next/image";

export function ChatMessageAvatar() {
  const { message } = useChatMessage();
  if (message.role === "user") {
    return (
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-background shadow">
        <User2 className="h-4 w-4" />
      </div>
    );
  }

  return (
    <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border text-white shadow">
      <Image
        className="rounded-md"
        src="/metanavit.jpeg"
        alt="MetaNaviT Logo"
        width={40}
        height={40}
        priority
      />
    </div>
  );
}
